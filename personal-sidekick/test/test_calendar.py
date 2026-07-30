import asyncio
import sys
sys.path.append("sidekick")
from agent import build_agent

async def main():
    agent = await build_agent()
    config = {"configurable": {"thread_id": "test-calendar-only"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Bugün hangi toplantılarım var?"}]},
        config=config,
    )
    print("Sidekick:", result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
