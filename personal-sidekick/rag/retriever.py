from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "sidekick_notes"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

client = QdrantClient(
    url="http://localhost:6333"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def search_notes(query: str, k: int = 5):
    """Kullanıcının kişisel notlarında semantik arama yapar."""

    results = vector_store.similarity_search(
        query=query,
        k=k,
    )

    return results