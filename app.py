import streamlit as st
import json
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AgriBot 🌾", layout="wide")

st.title("🌾 AgriBot — Farm Assistant")
st.caption("Chat about crops, soil, irrigation & pests • (Image disease detector coming next)")

# Load Groq API Key
GROQ_API_KEY = "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"

# ------------------ Load FAISS + Metadata ------------------ #
def load_faiss_vectorstore(embedding_model):
    index_path = "index.faiss"
    metadata_path = "metadata.json"

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found at {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata found at {metadata_path}")

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Rebuild docstore
    docstore = {k: Document(**v) for k, v in metadata["docstore"].items()}

    # Rebuild FAISS wrapper
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=metadata["index_to_docstore_id"]
    )
    return vectorstore


# ------------------ Usage ------------------ #
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = load_faiss_vectorstore(embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# ------------------ LLM (Groq) ------------------ #

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b"   # You can change to "mixtral-8x7b" or others
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def call_llm_direct(llm, user_msg, chat_history=None):
    """
    Fallback when retrieval fails: use LLM without FAISS.
    chat_history is optional and can be appended to the prompt.
    """
    system_prompt = (
        "You are an agronomy assistant. Be concise and helpful about crops, soil, irrigation, pests, and fertilizers."
    )
    history_txt = ""
    if chat_history:
        lines = []
        for u, a in chat_history[-6:]:
            lines.append(f"User: {u}\nAssistant: {a}")
        history_txt = "\n\n".join(lines)

    prompt = f"{system_prompt}\n\n"
    if history_txt:
        prompt += f"Conversation so far:\n{history_txt}\n\n"
    prompt += f"User: {user_msg}\nAssistant:"

    try:
        resp = llm.invoke(prompt)
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception:
        return "Sorry, I'm currently unable to answer your question."



# ------------------ Chatbot Section ------------------ #
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("💬 Chatbot")

    # Add CSS to keep chat input fixed at bottom
    st.markdown(
        """
        <style>
        /* Chat container scrollable */
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding-bottom: 100px; /* space for input bar */
        }
        /* Fixed input bar */
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: white;
            padding: 10px 20%;
            border-top: 1px solid #ddd;
            z-index: 999;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me about crop care, soil, irrigation, pests, or fertilizers."}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Scrollable chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ✅ FIXED input bar
st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
user_msg = st.chat_input("Type your question...")
st.markdown('</div>', unsafe_allow_html=True)

    # Render messages ABOVE input (reverse order)
    #with chat_container:
      #  for m in st.session_state.messages:
      #      with st.chat_message(m["role"]):
      #          st.write(m["content"])
        
    # Handle user input
    # Handle input
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    try:
        result = qa_chain({"question": user_msg, "chat_history": st.session_state.chat_history})
        reply = result["answer"]
    except Exception:
        reply = call_llm_direct(llm, user_msg, chat_history=st.session_state.chat_history)

    st.session_state.chat_history.append((user_msg, reply))
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

# ------------------ Disease Detector (Active Integration) ------------------ #
import numpy as np
from PIL import Image
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json

# Utility: Pre-process uploaded image for CNN model
def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load model and class indices only once (cache for efficiency)
@st.cache_resource
def load_cnn_model_and_labels():
    model = tf.keras.models.load_model("Plant_Disease_CNN_model.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Reverse mapping: numeric index -> class name
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_cnn_model_and_labels()

with right:
    st.subheader("🧪 Plant Disease Detector")
    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        class_label = labels[class_idx]
        confidence = float(np.max(prediction)) * 100

        st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
        st.success(f"*Predicted Disease:* {class_label}")
        st.info(f"*Confidence:* {confidence:.2f}%")

        # --- Send disease info to chatbot ---
        disease_query = f"I have detected '{class_label}' disease in my plant. What should I do?"
        if "disease_sent" not in st.session_state or st.session_state.disease_sent != class_label:
            st.session_state.disease_sent = class_label
            st.session_state.messages.append({"role": "user", "content": disease_query})
            with left:
                with st.chat_message("user"):
                    st.write(disease_query)
                with st.spinner("Thinking..."):
                    try:
                        result = qa_chain({"question": disease_query, "chat_history": st.session_state.chat_history})
                        reply = result["answer"]
                    except Exception as e:
                        reply = call_llm_direct(llm, disease_query, chat_history=st.session_state.chat_history)
                    st.session_state.chat_history.append((disease_query, reply))
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)
    else:
        st.info("Upload a leaf image to get a diagnosis.")
