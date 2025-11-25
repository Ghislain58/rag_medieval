Prérequis

Windows 10/11 avec WSL2 (Ubuntu)

Docker Desktop configuré avec backend WSL

GPU NVIDIA compatible CUDA + drivers à jour

NVIDIA Container Toolkit installé côté WSL (pour --gpus all)

Dans WSL, vérifier :

nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi

Démarrage rapide

Depuis WSL (Ubuntu) :

cd ~/rag_medieval

# 1. Construire l'image (GPU)
docker build --no-cache -t rag_medieval_app .

# 2. Réseau Docker pour RAG + Ollama
docker network create rag-net || true

# 3. Lancer Ollama (LLM local)
docker run --gpus all -d \
  --name ollama \
  --network rag-net \
  ollama/ollama

# 4. Télécharger le modèle Mistral dans Ollama
docker exec -it ollama ollama pull mistral

# 5. Lancer l'application RAG (Streamlit + Chroma)
docker run --gpus all -d \
  --name rag_app \
  --network rag-net \
  -p 8501:8501 \
  -v rag_data:/app/data \
  -e CHROMA_DIR="/app/data/chroma" \
  -e OLLAMA_URL="http://ollama:11434" \
  -e OLLAMA_MODEL="mistral" \
  rag_medieval_app


Interface disponible sur :
👉 http://localhost:8501

Utilisation de l’interface

L’UI propose 4 onglets :

📄 PDF

Upload d’un ou plusieurs PDF

Extraction texte + chunking + embeddings + indexation dans Chroma

🖼 Images

Upload d’images (png, jpg, jpeg, tif, tiff)

OCR (pytesseract, langues lat+fra) + indexation

🎙 Audio / vidéo

Upload de fichiers mp3, wav, m4a, mp4, mkv

Transcription via Whisper (small) + indexation

❓ Questions

Zone de texte pour la question historique

Le RAG récupère les extraits pertinents, construit un prompt,
appelle Mistral via Ollama et affiche la réponse argumentée,
avec les sources mobilisées (liste des fichiers utilisés).

Réinitialiser l’index

Pour repartir avec une base vectorielle vide :

docker stop rag_app || true
docker rm rag_app || true
docker volume rm rag_data || true


Puis relancer l’app (voir section “Démarrage rapide”).

Logs et debug

Afficher les logs de l’app :

docker logs rag_app


Afficher les logs d’Ollama (appel LLM) :

docker logs ollama
