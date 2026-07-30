from rag.rag_tool import search_my_notes

result = search_my_notes.invoke({
    "query": "MCP hakkında notlarımda ne öğrendim?"
})

print(result)