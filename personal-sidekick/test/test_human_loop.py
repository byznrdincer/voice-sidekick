import asyncio
from langgraph.types import Command
from sidekick.agent import build_agent

async def main():
    agent = await build_agent()

    config = {
        "configurable": {
            "thread_id": "hitl-mail-final"
        }
    }

    # 1 — Mail isteği
    result = await agent.ainvoke(
        {
            "messages": [{
                "role": "user",
                "content": (
                    "beyzanurdincer502@gmail.com adresine "
                    "'Human Loop Test' başlıklı "
                    "'Sidekick onay sistemi çalışıyor.' içerikli bir mail gönder."
                )
            }]
        },
        config=config,
    )

    print("\n=== ONAY BEKLENIYOR ===")
    print(result.get("__interrupt__"))

    # 2 — Kullanıcı onaylıyor
    result = await agent.ainvoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config,
    )

    print("\n=== ONAY SONRASI ===")
    print(result["messages"][-1].content)

asyncio.run(main())
