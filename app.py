"""app.py - Streamlit web page for the RAG chatbot. Run with: streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="Document Q&A Helper", page_icon="📚")
header = st.container()  # the title is filled in below, after the sidebar has run


def ask_about(topic):
    """Runs when a topic button is clicked: it fills the question box for you."""
    st.session_state["question"] = f"Explain {topic} in simple words, using the documents."


# ---- Sidebar: upload PDFs, build the index, and show the topics found ----
with st.sidebar:
    st.header("1. Add your documents")
    uploads = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Build index", disabled=not uploads):
        with st.spinner("Reading, chunking and embedding your PDFs..."):
            count = rag.build_index([(u.name, u) for u in uploads])
        if count:
            st.success(f"Done. Stored {count} chunks.")
        else:
            st.error("No text found. Scanned PDFs need OCR first.")

    headlines = rag.load_headlines()  # read after building, so new topics show straight away
    if headlines:
        st.header("Topics found")
        st.caption("Click a topic to ask about it.")
        for i, h in enumerate(headlines[:40]):
            st.button(
                f"p.{h['page']}  {h['text']}",
                key=f"topic{i}",
                on_click=ask_about,
                args=(h["text"],),
            )
        if len(headlines) > 40:
            st.caption(f"...and {len(headlines) - 40} more topics.")

# ---- Title at the top: it changes to match the documents you uploaded ----
with header:
    st.title(rag.load_title() or "Document Q&A Helper")
    st.caption("Upload any PDF notes, then ask questions. Answers come with the source page.")

# ---- Main area: ask questions ----
st.header("2. Ask a question")
if not rag.index_ready():
    st.info("Upload PDFs and click 'Build index' in the sidebar first.")
else:
    question = st.text_input(
        "Your question",
        key="question",
        placeholder="Type a question, or click a topic in the sidebar",
    )
    if question:
        with st.spinner("Searching your documents and asking the LLM..."):
            try:
                answer, chunks = rag.ask(question)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                st.stop()
        st.markdown(answer)
        with st.expander("Sources used"):
            for c in chunks:
                st.markdown(f"**{c['source']}, page {c['page']}**")
                st.write(c["text"])
