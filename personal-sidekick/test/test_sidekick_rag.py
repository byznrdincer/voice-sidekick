import asyncio
from sidekick.agent import build_agent

async def main():
    agent = await build_agent()

    config = {
        "configurable": {
            "thread_id": "rag-sidekick-test"
        }
    }

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Kendi staj notlarıma göre MCP nedir ve MCP'yi hangi konularda kullanmışım?"
                }
            ]
        },
        config=config,
    )

    print("\nSIDEKICK CEVABI:\n")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
