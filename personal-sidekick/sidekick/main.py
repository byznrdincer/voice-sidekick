"""
Faz 4 test ekranı — terminalden Sidekick ile sohbet (memory + tools + MCP).
Çalıştırmak için: python main.py
"""

import uuid
import asyncio
from agent import build_agent


async def run_chat():
    agent = await build_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Sidekick hazır (thread_id: {thread_id}). Çıkmak için 'exit' yaz.\n")

    while True:
        user_input = input("Sen: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Görüşürüz!")
            break
        if not user_input:
            continue

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        reply = result["messages"][-1].content
        print(f"Sidekick: {reply}\n")


if __name__ == "__main__":
    asyncio.run(run_chat())
