from langchain_core.tools import tool

from rag.retriever import search_notes


@tool
def search_my_notes(query: str) -> str:
    """
    Kullanıcının kişisel staj ve eğitim notlarında arama yapar.
    Kullanıcı kendi notları, öğrendiği konular veya staj günleri
    hakkında soru sorduğunda bu aracı kullan.
    """

    documents = search_notes(query)

    if not documents:
        return "Kişisel notlarda ilgili bilgi bulunamadı."

    results = []

    for doc in documents:
        source = doc.metadata.get("file_name", "Bilinmeyen dosya")

        results.append(
            f"""
Kaynak: {source}
İçerik:
{doc.page_content}
"""
        )

    return "\n---\n".join(results)