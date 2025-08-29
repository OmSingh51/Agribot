from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json
import numpy as np
from PIL import Image
import tensorflow as tf
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# -------- Config --------
GROQ_API_KEY = "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"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU
app = Flask(__name__)
CORS(app)

# -------- Load FAISS --------
def load_vectorstore(index_path="index.faiss", meta_path="metadata.json"):
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None
    index=faiss.read_index(index_path)
    meta=json.load(open(meta_path))
    docs={k:Document(**v) for k,v in meta["docstore"].items()}
    return FAISS(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                 index=index, docstore=docs, index_to_docstore_id=meta["index_to_docstore_id"])

vectordb=load_vectorstore()
retriever=vectordb.as_retriever(search_kwargs={"k":3}) if vectordb else None

# -------- Load LLM --------
llm=ChatGroq(groq_api_key=GROQ_API_KEY, model="openai/gpt-oss-20b")
qa_chain=ConversationalRetrievalChain.from_llm(llm,retriever) if retriever else None

# -------- Load CNN --------
try:
    model=tf.keras.models.load_model("Plant_Disease_CNN_model.h5")
    class_map=json.load(open("class_indices.json"))
    labels={int(v):k for k,v in class_map.items()}
except Exception as e:
    print("Model load error:",e)
    model=None; labels={}

# -------- Routes --------
@app.route("/chat", methods=["POST"])
def chat():
    msg=request.json.get("message","")
    if not msg: return jsonify({"error":"No message"}),400
    try:
        if qa_chain:
            result=qa_chain({"question":msg,"chat_history":[]})
            return jsonify({"reply":result.get("answer")})
    except: pass
    resp=llm.invoke(f"You are an agriculture assistant.\nUser:{msg}\nAssistant:")
    return jsonify({"reply":getattr(resp,"content",str(resp))})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files: return jsonify({"error":"No image"}),400
    if not model: return jsonify({"error":"Model not loaded"}),500
    img=Image.open(request.files["image"]).convert("RGB").resize((128,128))
    arr=np.expand_dims(np.array(img)/255.0,0)
    pred=model.predict(arr)
    idx=int(np.argmax(pred)); conf=float(np.max(pred))*100
    return jsonify({"class":labels.get(idx,str(idx)),"confidence":conf})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
