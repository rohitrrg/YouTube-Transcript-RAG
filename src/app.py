import streamlit as st
from youtube_rag import YouTubeRetriever

DEFAULT_LLM_MODEL = "MistralAI-7B"

@st.cache_resource
def get_retriever():
    return YouTubeRetriever()

ytr = get_retriever()

st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="wide")
st.title("🎬 YouTube RAG (LangChain + Streamlit)")
st.caption("Paste a YouTube URL. The app fetches the transcript, builds a vector index, and answers questions grounded in it.")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.markdown(f"**Model used:** `{DEFAULT_LLM_MODEL}`")
    chunk_size = st.slider("Chunk size", 500, 2500, 1200, step=100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 200, step=10)
    show_sources = st.checkbox("Show source chunks", value=False)

# --- Inputs ---
url = st.text_input(
    "YouTube URL (or 11-char video ID):",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo"
)
go = st.button("Load Transcript", type="primary")

# Session state
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = None


def _load_and_index(video_id: str):
    docs = ytr.load_transcript_as_docs(video_id)
    chunks = ytr.split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = ytr.vector_store_as_retriver(chunks)
    chain = ytr.chain(retriever)
    return retriever, chain


# Manual load
if go:
    try:
        vid = ytr.extract_video_id(url.strip())
        with st.spinner("Fetching transcript and building index…"):
            retriver, chain = _load_and_index(vid)
        st.success("Transcript loaded & indexed.")
        st.session_state.video_id = vid
        st.session_state.qa_chain = chain
        # st.session_state.loaded_docs = docs
    except Exception as e:
        st.error(f"Failed to load transcript: {e}")


st.divider()
st.subheader("Ask a question about the video")
question = st.text_input("Your question:", placeholder="e.g., Summarize the key takeaways.")

if st.button("Ask"):
    if not st.session_state.qa_chain:
        st.warning("Load a video first.")
    else:
        with st.spinner("Thinking…"):
            res = st.session_state.qa_chain.invoke(question)
        st.markdown("### Answer")
        st.write(res)

        if show_sources:
            srcs = res.get("source_documents", []) or []
            if srcs:
                with st.expander("Sources (retrieved chunks)"):
                    for i, d in enumerate(srcs, 1):
                        text = d.page_content.strip()
                        st.markdown(f"**Chunk {i}**")
                        st.write(text[:1200] + ("…" if len(text) > 1200 else ""))
            else:
                st.info("No source chunks returned.")
