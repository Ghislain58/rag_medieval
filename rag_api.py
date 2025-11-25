import os
import uuid
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import whisper
import pytesseract
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import torch

# ===============================
# 1. Chemins & répertoires
# ===============================

BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TRACES_DIR = DATA_DIR / "traces"

for p in (DATA_DIR, CHROMA_DIR, TRACES_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ===============================
# 2. Ressources partagées (GPU/CPU)
# ===============================

@st.cache_resource
def get_device() -> str:
    """Retourne 'cuda' si un GPU est visible, sinon 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Modèle d'embedding, bascule auto GPU/CPU."""
    device = get_device()
    print(f"[RAG] Chargement SentenceTransformer sur device={device}")
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)


@st.cache_resource
def load_whisper_model():
    """Modèle Whisper pour audio/vidéo."""
    device = get_device()
    print(f"[RAG] Chargement Whisper sur device={device}")
    return whisper.load_model("small", device=device)


@st.cache_resource
def get_chroma_collection():
    """Client ChromaDB persistant."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="medieval_corpus",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ===============================
# 3. Chunking texte
# ===============================

def chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """Chunking simple par caractères, avec recouvrement."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()

    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)

    return chunks


# ===============================
# 4. Ingestion dans ChromaDB
# ===============================

def _ingest_chunks(
    texts: List[str],
    source: str,
    doc_type: str,
    extra_meta: Dict[str, Any] | None = None,
) -> int:
    """Ajoute des chunks dans Chroma avec embeddings pré-calculés."""
    if not texts:
        return 0

    embedder = load_embedder()
    collection = get_chroma_collection()

    embeddings = embedder.encode(texts).tolist()

    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for i, t in enumerate(texts):
        meta: Dict[str, Any] = {
            "source": source,
            "doc_type": doc_type,
            "chunk_index": i,
        }
        if extra_meta:
            meta.update(extra_meta)
        metadatas.append(meta)
        ids.append(f"{doc_type}::{uuid.uuid4()}::chunk_{i}")

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(texts)


def add_pdf(uploaded_file) -> Tuple[bool, str]:
    """Indexe un PDF (extraction texte, chunking, embeddings, Chroma)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        texts: List[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            page_text = page_text.strip()
            if not page_text:
                continue
            page_chunks = chunk_text(page_text)
            texts.extend(page_chunks)

        if not texts:
            return False, "Aucun texte exploitable n'a été extrait de ce PDF."

        n = _ingest_chunks(
            texts,
            source=uploaded_file.name,
            doc_type="pdf",
        )
        return True, f"PDF '{uploaded_file.name}' indexé avec {n} chunk(s)."

    except Exception as e:
        return False, f"Erreur lors de l'indexation du PDF : {e}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def add_image(uploaded_file) -> Tuple[bool, str]:
    """Indexe une image (OCR fra+lat, chunking, embeddings, Chroma)."""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        ocr_text = pytesseract.image_to_string(image, lang="fra+lat").strip()

        if not ocr_text:
            return False, "Aucun texte détecté par OCR dans cette image."

        chunks = chunk_text(ocr_text)
        if not chunks:
            return False, "Le texte OCR est trop court ou vide après nettoyage."

        n = _ingest_chunks(
            chunks,
            source=uploaded_file.name,
            doc_type="image",
        )
        return True, f"Image '{uploaded_file.name}' indexée avec {n} chunk(s) OCR."

    except Exception as e:
        return False, f"Erreur lors de l'analyse de l'image : {e}"


def transcribe_audio(uploaded_file) -> str:
    """Transcrit un fichier audio/vidéo avec Whisper et indexe le texte."""
    model = load_whisper_model()

    tmp_path = None
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="fr")
        text = (result.get("text") or "").strip()

        if not text:
            return "Transcription vide ou non exploitable."

        chunks = chunk_text(text)
        if chunks:
            _ingest_chunks(
                chunks,
                source=uploaded_file.name,
                doc_type="audio",
            )

        return text

    except Exception as e:
        return f"Erreur lors de la transcription audio : {e}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ===============================
# 5. Vue d’ensemble & nettoyage index
# ===============================

def index_overview() -> Dict[str, Any]:
    """
    Résumé de l'index :
      - total_chunks
      - by_source : liste {source, doc_type, n_chunks}
    Compatible avec l'onglet "Index & nettoyage".
    """
    collection = get_chroma_collection()
    total = collection.count()

    by_key: Dict[Tuple[str, str], int] = {}

    # API Chroma récente : ne pas passer where={}
    res = collection.get(
        include=["metadatas"],
    )

    metadatas = res.get("metadatas", [])
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source", "inconnu")
        doc_type = meta.get("doc_type", "inconnu")
        key = (source, doc_type)
        by_key[key] = by_key.get(key, 0) + 1

    by_source = [
        {"source": src, "doc_type": dt, "n_chunks": n}
        for (src, dt), n in sorted(by_key.items(), key=lambda x: x[0][0])
    ]

    return {
        "total_chunks": total,
        "by_source": by_source,
    }


def delete_source_from_index(source: str) -> Dict[str, Any]:
    """
    Supprime tous les chunks dont meta['source'] == source.
    Retourne un petit rapport.
    """
    collection = get_chroma_collection()

    # API Chroma récente : where doit avoir un opérateur
    res = collection.get(
        where={"$and": [{"source": source}]},
        include=[],
    )
    ids = res.get("ids", []) or []

    deleted = 0
    if ids:
        deleted = len(ids)
        collection.delete(ids=ids)

    return {
        "status": "ok",
        "source": source,
        "deleted": deleted,
    }


# ===============================
# 6. Récupération RAG + LLM local
# ===============================

def _call_local_llm(prompt: str) -> str:
    """Appelle le modèle Ollama dans le service Docker 'ollama'."""
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "num_predict": 256,   # limite la longueur de réponse
            "temperature": 0.2,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un historien médiéviste critique et matérialiste. "
                    "Tu NE DOIS JAMAIS inventer ou extrapoler. "
                    "Tu ne peux répondre QUE sur la base des extraits fournis. "
                    "Si l'information n'est pas présente dans les extraits, tu écris : "
                    "'Les extraits fournis ne permettent pas de répondre précisément.' "
                    "Tu restes concis, factuel, et tu signales clairement les limites "
                    "documentaires et les hypothèses."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content") or ""
        if not content:
            return "[LLM local] Réponse vide ou inattendue."
        return content.strip()
    except Exception as e:
        return f"[LLM local] Impossible d'appeler le modèle ({e})."


def _retrieve_top_k(query: str, k: int = 5) -> Dict[str, Any]:
    """Interroge Chroma avec embedding de la requête."""
    collection = get_chroma_collection()
    embedder = load_embedder()

    query_emb = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def _build_prompt(query: str, results: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Construit le prompt pour le LLM + la liste de chunks structurés
    (pour la trace JSON et le résumé des sources).
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    chunks_for_trace: List[Dict[str, Any]] = []
    context_parts: List[str] = []

    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        source = meta.get("source", "inconnu")
        doc_type = meta.get("doc_type", "inconnu")
        page = meta.get("page")
        score = 1.0 - float(dist) if dist is not None else None

        header = f"[source={source} | type={doc_type}"
        if page is not None:
            header += f" | page={page}"
        if score is not None:
            header += f" | score_approx={score:.3f}"
        header += "]"

        context_parts.append(header)
        context_parts.append(doc)
        context_parts.append("")

        chunks_for_trace.append(
            {
                "text": doc,
                "source": source,
                "doc_type": doc_type,
                "page": page,
                "score_approx": score,
            }
        )

    context_block = "\n".join(context_parts).strip()

    prompt = f"""
Tu es un historien médiéviste critique.
Tu dois répondre à la question en t'appuyant UNIQUEMENT sur les extraits suivants.

### Extraits fournis
{context_block}

### Question de recherche
{query}

### Consignes
- Si les sources ne permettent pas de répondre, dis-le explicitement.
- Signale les limites et les lacunes documentaires.
- Appuie ta réponse sur les formulations des textes, sans inventer.
- Contextualise socialement et politiquement, sans anachronisme.
- Reste concis et structuré.

### Réponse structurée
"""
    return prompt.strip(), chunks_for_trace


# ===============================
# 7. Trace JSON historique
# ===============================

def _save_trace(
    query: str,
    chunks: List[Dict[str, Any]],
    answer_text: str,
) -> None:
    """Sauvegarde une trace JSON de la requête et de la réponse."""
    now = datetime.utcnow().isoformat()
    safe_ts = now.replace(":", "").replace(".", "_")
    trace_path = TRACES_DIR / f"trace_{safe_ts}.json"

    trace = {
        "timestamp_utc": now,
        "query": query,
        "answer": answer_text,
        "chunks_used": chunks,
    }

    try:
        with trace_path.open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[TRACE] Impossible d'écrire la trace JSON : {e}")


# ===============================
# 8. API principale pour Streamlit
# ===============================

def answer(query: str) -> str:
    """
    Pipeline complet :
      - récupère top-k dans Chroma
      - construit prompt
      - appelle Ollama
      - enregistre trace JSON
      - renvoie une réponse Markdown
    """
    results = _retrieve_top_k(query, k=5)
    if not results.get("documents") or not results["documents"][0]:
        return "Aucun document indexé n'a pu être retrouvé pour cette question."

    prompt, chunks_for_trace = _build_prompt(query, results)
    llm_answer = _call_local_llm(prompt)
    _save_trace(query, chunks_for_trace, llm_answer)

    # Déduplication des sources pour l'affichage
    seen_keys = set()
    unique_sources: List[Dict[str, Any]] = []
    for c in chunks_for_trace:
        key = (c.get("source"), c.get("doc_type"), c.get("page"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_sources.append(c)

    sources_md_lines: List[str] = []
    for c in unique_sources:
        src = c.get("source", "inconnu")
        doc_type = c.get("doc_type", "inconnu")
        page = c.get("page")
        score = c.get("score_approx")

        line = f"- **{src}** (*{doc_type}*)"
        if page is not None:
            line += f", p. {page}"
        if score is not None:
            line += f" — score ~ {score:.3f}"
        sources_md_lines.append(line)

    sources_md = "\n".join(sources_md_lines)

    return (
        llm_answer
        + "\n\n---\n"
        + "**Sources mobilisées (top-k, dédupliquées)** :\n"
        + sources_md
    )
