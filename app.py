
"""
Neomorphic RAG Studio

Run with: streamlit run app.py
"""

import os
import shutil
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Studio",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
NEO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --cream: #efe6dd;
    --cream-deep: #e6dacc;
    --cherry: #9a0002;
    --cherry-deep: #740002;
    --cherry-bright: #c4161a;
    --shadow-dark: #cfc2b3;
    --shadow-light: #ffffff;
    --text-main: #111111;
    --text-sub: #7a6a5c;
    --on-cherry: #f6e9d9;
    --success: #3f7a4e;
    --radius: 20px;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f4ede4 0%, #ecdfd0 100%);
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* ---------- Containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--cream) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    box-shadow:
        9px 9px 16px var(--shadow-dark),
        -9px -9px 16px var(--shadow-light) !important;
    padding: 0.4rem 0.3rem !important;
    margin-bottom: 1.1rem !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--cream) !important;
    box-shadow:
        6px 6px 12px var(--shadow-dark),
        -6px -6px 12px var(--shadow-light) !important;
}

/* ---------- Header ---------- */
.app-title {
    font-weight: 800;
    font-size: 2.35rem;
    letter-spacing: -0.5px;
    color: var(--cherry);
    margin-bottom: 0.1rem;
}

.app-subtitle {
    font-weight: 500;
    color: var(--text-sub);
    font-size: 1.02rem;
    margin-bottom: 1.4rem;
}

.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--cherry);
    padding: 7px 18px;
    border-radius: 999px;
    box-shadow:
        5px 5px 10px var(--shadow-dark),
        -5px -5px 10px var(--shadow-light);
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    color: var(--on-cherry);
    margin-bottom: 1.1rem;
}

.section-label {
    font-weight: 700;
    font-size: 0.76rem;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: var(--cherry);
    margin: 0.9rem 0 0.5rem 0.2rem;
}

/* ---------- Status badges ---------- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    background: var(--cream);
    box-shadow:
        4px 4px 8px var(--shadow-dark),
        -4px -4px 8px var(--shadow-light);
}

.badge-ready { color: var(--success); }
.badge-wait { color: var(--text-sub); }

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--cream);
    color: var(--text-main);
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    font-size: 0.92rem;
    box-shadow:
        6px 6px 12px var(--shadow-dark),
        -6px -6px 12px var(--shadow-light);
    transition: all 0.15s ease;
}

.stButton > button:hover {
    color: var(--cherry);
    box-shadow:
        3px 3px 6px var(--shadow-dark),
        -3px -3px 6px var(--shadow-light);
}

.stButton > button:active {
    box-shadow:
        inset 4px 4px 8px var(--shadow-dark),
        inset -4px -4px 8px var(--shadow-light);
}

[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: var(--cherry) !important;
    color: var(--on-cherry) !important;
    border: none !important;
    box-shadow:
        4px 4px 8px var(--shadow-dark),
        -4px -4px 8px var(--shadow-light) !important;
}

[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="primary"] span,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] span {
    color: var(--on-cherry) !important;
}

[data-testid="stSidebar"] button[kind="primary"]:disabled,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:disabled {
    opacity: 0.85 !important;
    cursor: not-allowed;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--cream-deep) !important;
    border-radius: 16px !important;
    border: 2px dashed rgba(154, 0, 2, 0.3) !important;
    box-shadow:
        inset 5px 5px 10px var(--shadow-dark),
        inset -5px -5px 10px var(--shadow-light);
}

[data-testid="stFileUploaderDropzone"] :is(p, span, small) {
    color: var(--text-main) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--cherry) !important;
    color: var(--on-cherry) !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow:
        4px 4px 8px var(--shadow-dark),
        -4px -4px 8px var(--shadow-light);
}

[data-testid="stFileUploaderDropzone"] button :is(p, span) {
    color: var(--on-cherry) !important;
}

[data-testid="stFileUploaderDropzone"] button svg {
    color: var(--on-cherry) !important;
    fill: currentColor !important;
}

/* ---------- Main text and chat messages ---------- */
[data-testid="stMain"] {
    color: var(--text-main);
}

[data-testid="stChatMessage"] {
    background: var(--cream) !important;
    color: var(--text-main) !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow:
        6px 6px 12px var(--shadow-dark),
        -6px -6px 12px var(--shadow-light) !important;
}

/* Text only: do not force SVG path fills, which can distort icons. */
[data-testid="stChatMessage"] :is(
    p, span, small, strong, b, em, i, a, li,
    h1, h2, h3, h4, h5, h6,
    blockquote, code, pre, summary
) {
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: var(--cherry) !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--cherry-deep) !important;
}

/* ---------- Expanders and retrieved context ---------- */
[data-testid="stExpander"] {
    background: var(--cream-deep) !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow:
        inset 3px 3px 6px var(--shadow-dark),
        inset -3px -3px 6px var(--shadow-light);
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary :is(p, span) {
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
}

[data-testid="stExpander"] summary svg {
    color: var(--text-main) !important;
}

.source-box {
    background: var(--cream-deep);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-main);
    box-shadow:
        inset 4px 4px 8px var(--shadow-dark),
        inset -4px -4px 8px var(--shadow-light);
    margin-bottom: 0.6rem;
    max-height: 140px;
    overflow-y: auto;
}

.source-box b {
    color: var(--text-main);
}

/* ---------- Bottom chat dock---------- */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stChatFloatingInputContainer"] {
    background: var(--cherry) !important;
    background-image: none !important;
}

/* Remove Streamlit's default fade/gradient over the dock. */
[data-testid="stBottom"]::before,
[data-testid="stChatFloatingInputContainer"]::before,
[data-testid="stChatFloatingInputContainer"]::after {
    background: transparent !important;
    background-image: none !important;
}

/* Chat input itself stays cream, with black text. */
[data-testid="stChatInput"] {
    background: var(--cream) !important;
    border: 2px solid var(--cherry-deep) !important;
    border-radius: 18px !important;
    box-shadow:
        inset 5px 5px 10px var(--shadow-dark),
        inset -5px -5px 10px var(--shadow-light) !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    background: var(--cream) !important;
    border-radius: 18px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--cherry-bright) !important;
    box-shadow: 0 0 0 2px rgba(154, 0, 2, 0.3) !important;
}

[data-testid="stChatInput"] textarea {
    background: var(--cream) !important;
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
    caret-color: var(--cherry) !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: var(--cherry) !important;
    color: var(--on-cherry) !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] button:hover:not(:disabled) {
    background: var(--cherry-bright) !important;
}

[data-testid="stChatInput"] button svg {
    color: var(--on-cherry) !important;
}

[data-testid="stChatInput"] button:focus-visible {
    outline: 2px solid var(--cherry-bright) !important;
}

/* ---------- Other inputs ---------- */
.stTextInput > div > div > input {
    background: var(--cream-deep) !important;
    color: var(--text-main) !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow:
        inset 5px 5px 10px var(--shadow-dark),
        inset -5px -5px 10px var(--shadow-light) !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7a0002, #5c0001);
    border-right: 1px solid rgba(0, 0, 0, 0.15);
}

[data-testid="stSidebar"] .section-label {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] :is(p, span, label),
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: var(--cream) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: var(--cherry) !important;
    border-color: var(--cherry) !important;
}

[data-testid="stSlider"] [data-testid="stSliderTrackColor"] {
    background: var(--cherry) !important;
}

/* ---------- Model selectbox ---------- */
/* Give the whole widget a light surface so its black label stays readable. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    background: var(--cream) !important;
    border-radius: 14px !important;
    padding: 0.35rem !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label :is(p, span) {
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--cream-deep) !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow:
        inset 4px 4px 8px var(--shadow-dark),
        inset -4px -4px 8px var(--shadow-light) !important;
}

/* Selected model text. These rules do not target SVG elements. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] :is(div, span, input) {
    color: var(--text-main) !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div:not(:has(svg)) {
    -webkit-text-fill-color: var(--text-main) !important;
}

/* Color the arrow through currentColor; do not force its paths or stroke. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: var(--text-main) !important;
    fill: currentColor !important;
    background: transparent !important;
}

[data-baseweb="popover"] [role="listbox"] {
    background: var(--cream) !important;
}

[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] :is(div, span) {
    color: var(--text-main) !important;
}

/* ---------- Stats and welcome card ---------- */
.stat-chip {
    background: var(--cream);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    box-shadow:
        5px 5px 10px var(--shadow-dark),
        -5px -5px 10px var(--shadow-light);
    text-align: center;
}

.stat-value {
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--cherry-deep);
    overflow-wrap: anywhere;
}

.stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-sub);
    font-weight: 700;
    margin-top: 2px;
}

.neo-divider {
    height: 2px;
    background: linear-gradient(
        90deg,
        transparent,
        var(--shadow-dark),
        transparent
    );
    opacity: 0.7;
    margin: 1.1rem 0;
    border-radius: 2px;
}

.welcome-card {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.welcome-card,
.welcome-card * {
    color: var(--text-main) !important;
}

.welcome-emoji {
    font-size: 2.4rem;
    line-height: 1;
}

/* ---------- Progress bar ---------- */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        var(--cherry-deep),
        var(--cherry-bright)
    ) !important;
}
</style>
"""

st.markdown(NEO_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "vectorstore": None,
    "retriever": None,
    "messages": [],
    "doc_name": None,
    "chunk_count": 0,
    "page_count": 0,
    "index_time": None,
    "persist_dir": None,
}

for _key, _value in defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    return ChatGroq(model=model_name)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def build_index(uploaded_file, chunk_size: int, chunk_overlap: int, progress_cb=None):
    """Save the PDF, split it, embed its chunks, and create a Chroma store."""
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    pdf_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if progress_cb:
        progress_cb(0.15, "Reading PDF pages…")

    docs = PyPDFLoader(pdf_path).load()

    if progress_cb:
        progress_cb(0.35, "Splitting into chunks…")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    if progress_cb:
        progress_cb(0.55, "Loading embedding model…")

    embedding_model = get_embedding_model()
    persist_dir = tempfile.mkdtemp(prefix="chroma_")

    if progress_cb:
        progress_cb(0.75, "Embedding chunks into vector store…")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )

    if progress_cb:
        progress_cb(1.0, "Index ready.")

    return vectorstore, len(chunks), len(docs), persist_dir


def get_retriever(vectorstore, k: int, fetch_k: int, lambda_mult: float):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def answer_question(retriever, llm, query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-pill">🍒&nbsp; RAG STUDIO</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Document</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        with st.expander("⚙️ Indexing settings", expanded=False):
            chunk_size = st.slider(
                "Chunk size", 300, 2000, 1000, step=100
            )
            chunk_overlap = st.slider(
                "Chunk overlap", 0, 500, 200, step=50
            )

        build_clicked = st.button(
            "🚀 Build index",
            use_container_width=True,
            type="primary",
            disabled=uploaded_file is None,
        )

    if build_clicked and uploaded_file is not None:
        progress_bar = st.progress(0, text="Starting…")

        def update_progress(pct, label):
            progress_bar.progress(pct, text=label)

        try:
            start = time.time()
            vectorstore, n_chunks, n_pages, persist_dir = build_index(
                uploaded_file,
                chunk_size,
                chunk_overlap,
                progress_cb=update_progress,
            )
            elapsed = time.time() - start

            old_dir = st.session_state.persist_dir
            if old_dir and os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)

            st.session_state.vectorstore = vectorstore
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chunk_count = n_chunks
            st.session_state.page_count = n_pages
            st.session_state.index_time = round(elapsed, 1)
            st.session_state.persist_dir = persist_dir
            st.session_state.messages = []

            st.success(
                f"Indexed **{uploaded_file.name}** in {elapsed:.1f}s"
            )
        except Exception as exc:
            st.error(f"Failed to build index: {exc}")

    st.markdown(
        '<div class="section-label">Retrieval</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        k = st.slider("Top-k results", 1, 10, 4)
        fetch_k = st.slider(
            "Fetch-k (MMR pool)", k, 20, max(10, k)
        )
        lambda_mult = st.slider(
            "MMR diversity (λ)", 0.0, 1.0, 0.5
        )

    st.markdown(
        '<div class="section-label">Model</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        model_name = st.selectbox(
            "Groq model",
            [
                "openai/gpt-oss-20b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
            index=0,
        )

    if st.session_state.vectorstore is not None:
        st.markdown(
            '<div class="section-label">Session</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.caption(f"📄 {st.session_state.doc_name}")

            if st.button("🗑️ Clear index", use_container_width=True):
                old_dir = st.session_state.persist_dir
                if old_dir and os.path.isdir(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)

                for key in defaults:
                    st.session_state[key] = defaults[key]

                st.rerun()


# --------------------------------------------------------------------------
# Main header
# --------------------------------------------------------------------------
left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<div class="app-title">Document Q&amp;A</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">'
        'Upload a PDF, build a vector index, and ask questions '
        'grounded strictly in its content.'
        '</div>',
        unsafe_allow_html=True,
    )

with right:
    if st.session_state.vectorstore is not None:
        st.markdown(
            '<div class="badge badge-ready" style="margin-top:1.6rem;">'
            '● Index ready'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge badge-wait" style="margin-top:1.6rem;">'
            '○ No document indexed'
            '</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    c1, c2, c3, c4 = st.columns(4)

    stats = [
        (c1, "Document", st.session_state.doc_name or "—"),
        (c2, "Pages", st.session_state.page_count),
        (c3, "Chunks", st.session_state.chunk_count),
        (c4, "Indexed in", f"{st.session_state.index_time}s"),
    ]

    for column, label, value in stats:
        with column:
            st.markdown(
                f'<div class="stat-chip">'
                f'<div class="stat-value">{value}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="neo-divider"></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
if st.session_state.vectorstore is None:
    with st.container(border=True):
        st.markdown(
            '<div class="welcome-card">'
            '<div class="welcome-emoji">👋</div>'
            '<div><b>Get started</b> — upload a PDF from the sidebar '
            'and click <b>Build index</b>. Once indexing finishes, '
            'you can ask questions about the document right here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View retrieved context"):
                    for index, source in enumerate(
                        message["sources"], start=1
                    ):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>Chunk {index}</b><br>{source}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask something about your document…")

    if query:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    retriever = get_retriever(
                        st.session_state.vectorstore,
                        k,
                        fetch_k,
                        lambda_mult,
                    )
                    llm = get_llm(model_name)
                    answer, docs = answer_question(
                        retriever, llm, query
                    )
                    sources = [doc.page_content for doc in docs]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📄 View retrieved context"):
                            for index, source in enumerate(
                                sources, start=1
                            ):
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<b>Chunk {index}</b><br>{source}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    error = f"⚠️ Something went wrong: {exc}"
                    st.error(error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error}
                    )
