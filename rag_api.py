import os
import uuid
import tempfile
from typing import List, Dict, Any

import streamlit as st
import whisper
import pytesseract
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")


class RAGEngine:
    def __init__(self) -> None:
        self.embedder = load_embedder()
        self.whisper_model = load_whisper_model()

        # Dossier de persistance Chroma (sur disque)
        persist_dir = os.getenv("CHROMA_DIR", "data/chroma")
        os.makedirs(persist_dir, exist_ok=True)

        # NOUVELLE API : client persistant
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Collection vectorielle persistante
        self.collection = self.client.get_or_create_collection(
            name="historical_rag"
        )

    @staticmethod
    def _chunk_text(
        text: str,
        max_chars: int = 1000,
        overlap: int = 200,
    ) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break

            start = max(0, end - overlap)

        return chunks

    def add_document(
        self,
        text: str,
        source: str,
        doc_type: str = "pdf",
    ) -> None:
        text = (text or "").strip()
        if not text:
            return

        chunks = self._chunk_text(text)
        if not chunks:
            return

        embeddings = self.embedder.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source, "doc_type": doc_type} for _ in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        q_emb = self.embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if q_emb.ndim > 1:
            q_emb = q_emb[0]

        res = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
        )

        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        dists_list = res.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            meta = meta or {}
            source = meta.get("source", "inconnu")
            doc_type = meta.get("doc_type", "unknown")
            try:
                score = float(dist) if dist is not None else None
            except Exception:
                score = None

            results.append(
                {
                    "score": score,
                    "text": doc,
                    "source": source,
                    "doc_type": doc_type,
                }
            )

        return results

    def transcribe_audio_file(self, file) -> str:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            result = self.whisper_model.transcribe(tmp_path, language="fr")
            text = result.get("text", "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return text


@st.cache_resource
def get_engine() -> RAGEngine:
    return RAGEngine()


def add_pdf(file_obj) -> None:
    engine = get_engine()
    reader = PdfReader(file_obj)

    pages_text: List[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue

    full_text = "\n\n".join(pages_text)
    source_name = getattr(file_obj, "name", "pdf")
    engine.add_document(full_text, source=source_name, doc_type="pdf")


def add_image(file_obj) -> None:
    engine = get_engine()
    image = Image.open(file_obj)
    text = pytesseract.image_to_string(image, lang="lat+fra")
    source_name = getattr(file_obj, "name", "image")
    engine.add_document(text, source=source_name, doc_type="image")


def transcribe_audio(file_obj) -> str:
    engine = get_engine()
    text = engine.transcribe_audio_file(file_obj)
    source_name = getattr(file_obj, "name", "audio")
    engine.add_document(text, source=source_name, doc_type="audio")
    return text


def _build_llm_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_blocks: List[str] = []
    for i, item in enumerate(retrieved, start=1):
        source = item["source"]
        text = (item["text"] or "").strip().replace("\n", " ")
        if len(text) > 900:
            text = text[:900] + " …"
        block = f"[{i}] (source : {source})\n{text}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    prompt = (
        "Tu es un historien médiéviste critique. "
        "Tu travailles sur des sources primaires (chartes, actes, études érudites). "
        "Réponds en français, de manière argumentée, en citant les extraits par leurs numéros entre crochets.\n\n"
        f"Question de l'utilisateur :\n{question}\n\n"
        "Extraits disponibles :\n"
        f"{context_str}\n\n"
        "Consignes :\n"
        "- Appuie-toi uniquement sur ces extraits.\n"
        "- Mentionne explicitement les numéros des extraits utilisés (ex : [1], [3]…).\n"
        "- Signale clairement ce qui relève de l'hypothèse ou de l'interprétation.\n"
        "- Termine par une courte synthèse (3–4 phrases) qui replace la réponse dans un contexte plus large."
    )
    return prompt


def _call_local_llm(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un historien médiéviste critique, matérialiste, rigoureux sur les sources, "
                    "capable de contextualiser socialement, politiquement et économiquement les documents."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            return "[LLM local] Réponse vide ou inattendue.\n\n" + prompt
        return content.strip()
    except Exception as e:
        return (
            f"[LLM local] Impossible d'appeler le modèle ({e}).\n\n"
            "Je te renvoie le prompt de contexte pour diagnostic :\n\n"
            f"{prompt}"
        )


def answer(query: str) -> str:
    engine = get_engine()
    retrieved = engine.retrieve(query, top_k=7)

    if not retrieved:
        return (
            "### Aucune source disponible\n\n"
            "Je n'ai pour l'instant aucun document indexé. "
            "Ajoute des PDF / images / audios dans les autres onglets, puis repose ta question."
        )

    prompt = _build_llm_prompt(query, retrieved)
    llm_answer = _call_local_llm(prompt)

    try:
        total_chunks = engine.collection.count()
    except Exception:
        total_chunks = None

    unique_sources: List[str] = []
    for item in retrieved:
        src = item["source"]
        if src not in unique_sources:
            unique_sources.append(src)

    sources_md = "\n".join(f"- {src}" for src in unique_sources)

    header = ""
    if total_chunks is not None:
        header = (
            f"**Index actuel** : {total_chunks} extrait(s) textuel(s) indexé(s).\n\n"
            "---\n\n"
        )

    return (
        header
        + llm_answer
        + "\n\n---\n"
        + "**Sources mobilisées (top-k)** :\n"
        + sources_md
    )
