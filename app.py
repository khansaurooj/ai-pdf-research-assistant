from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
import uuid

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { background-color: #f7f8fa; }
    .stApp { font-family: 'Inter', sans-serif; }
    h1 { font-weight: 700; color: #1a1a2e; }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }
    .chat-bubble-user {
        background-color: #4f46e5; color: white; padding: 12px 16px;
        border-radius: 16px 16px 4px 16px; margin: 8px 0; max-width: 80%;
        margin-left: auto; text-align: right;
    }
    .chat-bubble-assistant {
        background-color: white; color: #1a1a2e; padding: 12px 16px;
        border-radius: 16px 16px 16px 4px; margin: 8px 0; max-width: 80%;
        border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .source-card {
        background-color: #f9fafb; border-left: 3px solid #4f46e5;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0; font-size: 0.85rem;
    }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .chat-history-item {
        padding: 8px 10px; border-radius: 6px; margin: 4px 0;
        font-size: 0.85rem; cursor: pointer;
    }
    .chat-history-active {
        background-color: #eef2ff; border-left: 3px solid #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load models once ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---- Backend functions ----
def extract_pdf(path):
    doc = fitz.open(path)
    pages = [{"page": i+1, "text": p.get_text()} for i, p in enumerate(doc)]
    doc.close()
    return pages

def chunk_text(pages, chunk_size=500, overlap=50):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": p["page"], "text": piece})
            start += chunk_size - overlap
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query, index, chunks, top_k=3):
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({**chunks[idx], "score": float(dist)})
    return results

def generate_answer(query, retrieved_chunks, history=[]):
    context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks])
    prompt = f"""You are an AI Research Assistant.
Answer ONLY using the provided context below.
If the answer is not in the context, reply exactly: "I couldn't find that information in the uploaded documents."

Context:
{context}

Question: {query}
"""
    messages = history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

# ---- Session state ----
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pages_count" not in st.session_state:
    st.session_state.pages_count = 0

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New Chat", "messages": [], "last_results": []}
    st.session_state.active_chat_id = chat_id
    return chat_id

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None and st.session_state.index is None:
        with st.spinner("Processing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pages = extract_pdf("temp.pdf")
            chunks = chunk_text(pages)
            index = build_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.pages_count = len(pages)
        st.success("Document ready!")

    if st.session_state.index is not None:
        st.markdown("---")

        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.markdown("### 🕒 Chat History")
        if not st.session_state.chats:
            st.caption("No chats yet — click 'New Chat' to start.")
        else:
            for cid, chat in reversed(list(st.session_state.chats.items())):
                is_active = cid == st.session_state.active_chat_id
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    label = ("🟢 " if is_active else "") + chat["title"]
                    if st.button(label, key=f"select_{cid}", use_container_width=True):
                        st.session_state.active_chat_id = cid
                        st.rerun()
                with col_b:
                    if st.button("🗑️", key=f"delete_{cid}"):
                        del st.session_state.chats[cid]
                        if st.session_state.active_chat_id == cid:
                            st.session_state.active_chat_id = None
                        st.rerun()

        st.markdown("---")
        if st.button("🔄 New Document", use_container_width=True):
            st.session_state.index = None
            st.session_state.chunks = None
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            st.rerun()

    st.markdown("---")
    st.caption("Built with FAISS · Sentence-Transformers · Groq (Llama 3.3) · Streamlit")

# ---- Main area ----
st.markdown("# 📄 AI PDF Research Assistant")
st.markdown('<p class="subtitle">Ask questions grounded in your document — every answer cites its source page.</p>', unsafe_allow_html=True)

if st.session_state.index is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    if st.session_state.active_chat_id is None:
        create_new_chat()

    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    for msg in active_chat["messages"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input("Ask a question", placeholder="What is this document about?", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and query:
        with st.spinner("Thinking..."):
            results = retrieve(query, st.session_state.index, st.session_state.chunks)
            answer = generate_answer(query, results, active_chat["messages"])
            active_chat["messages"].append({"role": "user", "content": query})
            active_chat["messages"].append({"role": "assistant", "content": answer})
            active_chat["last_results"] = results
            if active_chat["title"] == "New Chat":
                active_chat["title"] = query[:35] + ("..." if len(query) > 35 else "")
        st.rerun()

    if active_chat["last_results"]:
        with st.expander("📚 Sources used in last answer"):
            for r in active_chat["last_results"]:
                st.markdown(f"""<div class="source-card"><b>Page {r['page']}</b> · similarity score: {r['score']:.3f}<br>{r['text'][:200]}...</div>""", unsafe_allow_html=True)