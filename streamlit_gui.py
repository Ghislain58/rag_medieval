import streamlit as st
from rag_api import add_pdf, add_image, transcribe_audio, answer

st.set_page_config(
    page_title="RAG médiéval local",
    page_icon="📜",
    layout="wide",
)

st.title("📜 RAG médiéval local (Docker + GPU + ChromaDB)")
st.markdown(
    """
Ce prototype indexe **PDF**, **images (OCR)** et **audio/vidéo (Whisper)** dans une base vectorielle locale
(ChromaDB), puis utilise un **LLM local** (Ollama) pour répondre aux questions avec un regard d'historien
médiéviste critique.
"""
)

tab_pdf, tab_img, tab_audio, tab_q = st.tabs(
    ["📄 PDF", "🖼 Images", "🎙 Audio / vidéo", "❓ Questions"]
)

# -----------------------------
# Onglet PDF
# -----------------------------
with tab_pdf:
    st.header("Importer des PDF")
    pdf_files = st.file_uploader(
        "Sélectionne un ou plusieurs fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if pdf_files:
        for f in pdf_files:
            with st.spinner(f"Indexation de {f.name} ..."):
                add_pdf(f)
            st.success(f"{f.name} indexé.")

# -----------------------------
# Onglet Images
# -----------------------------
with tab_img:
    st.header("Importer des images (chartes, scans, etc.)")
    img_files = st.file_uploader(
        "Sélectionne une ou plusieurs images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if img_files:
        for f in img_files:
            with st.spinner(f"OCR + indexation de {f.name} ..."):
                add_image(f)
            st.success(f"{f.name} indexée.")

# -----------------------------
# Onglet Audio / vidéo
# -----------------------------
with tab_audio:
    st.header("Importer des fichiers audio ou vidéo")
    audio_files = st.file_uploader(
        "Sélectionne un ou plusieurs fichiers audio/vidéo",
        type=["mp3", "wav", "m4a", "mp4", "mkv"],
        accept_multiple_files=True,
    )
    if audio_files:
        for f in audio_files:
            with st.spinner(f"Transcription + indexation de {f.name} ..."):
                txt = transcribe_audio(f)
            st.success(f"{f.name} transcrit et indexé.")
            with st.expander(f"Transcription de {f.name}"):
                st.write(txt)

# -----------------------------
# Onglet Questions
# -----------------------------
with tab_q:
    st.header("Poser une question au corpus indexé")
    query = st.text_area(
        "Formule ta question historique (contexte, période, acteurs…) :",
        height=120,
    )
    if st.button("Lancer la recherche"):
        if not query.strip():
            st.warning("Merci de saisir une question.")
        else:
            with st.spinner("Recherche dans l'index vectoriel + appel du LLM local..."):
                response = answer(query.strip())
            st.markdown(response)
