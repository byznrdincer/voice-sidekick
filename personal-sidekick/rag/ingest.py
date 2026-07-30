import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

DATA_DIR = Path("data/notes")
COLLECTION_NAME = "sidekick_notes"

# 1. PDF'leri oku
documents = []

for pdf_path in DATA_DIR.glob("*.pdf"):
    print(f"Okunuyor: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # Hangi nottan geldiğini metadata'da saklayalım
    for doc in docs:
        doc.metadata["file_name"] = pdf_path.name

    documents.extend(docs)

print(f"Toplam sayfa: {len(documents)}")

# 2. Chunk'lara böl
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Toplam chunk: {len(chunks)}")

# 3. Embedding modeli
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 4. Qdrant bağlantısı
client = QdrantClient(
    url="http://localhost:6333"
)

# 5. Collection'ı temiz oluştur
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
    ),
)

# 6. Chunk'ları embedding'e çevir ve Qdrant'a kaydet
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

vector_store.add_documents(chunks)

print("\nRAG ingest tamamlandı.")
print(f"{len(chunks)} chunk Qdrant'a kaydedildi.")