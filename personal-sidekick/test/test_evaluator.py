import asyncio

from sidekick.agent import build_agent
from sidekick.evaluator import evaluate_result


async def run_sidekick(task, success_criteria, max_attempts=3):
    agent = await build_agent()

    config = {
        "configurable": {
            "thread_id": "evaluator-test"
        }
    }

    message = task

    for attempt in range(1, max_attempts + 1):

        print(f"\n===== DENEME {attempt} =====")

        result = await agent.ainvoke(
            {
                "messages": [{
                    "role": "user",
                    "content": message
                }]
            },
            config=config,
        )

        # Human-in-the-loop oluşursa burada dur.
        if result.get("__interrupt__"):
            print("\nONAY GEREKİYOR:")
            print(result["__interrupt__"])
            return

        answer = result["messages"][-1].content

        print("\nSIDEKICK:")
        print(answer)

        evaluation = await evaluate_result(
            task=task,
            success_criteria=success_criteria,
            answer=answer,
        )

        print("\nEVALUATOR:")
        print("Success:", evaluation.success)
        print("Feedback:", evaluation.feedback)

        if evaluation.success:
            print("\nGÖREV BAŞARIYLA TAMAMLANDI.")
            return answer

        message = f"""
Önceki çalışman başarı kriterlerini karşılamadı.

Evaluator feedback:
{evaluation.feedback}

Orijinal görev:
{task}

Başarı kriterleri:
{success_criteria}

Eksikleri düzelt ve görevi yeniden tamamla.
"""

    print("\nMaksimum deneme sayısına ulaşıldı.")


async def main():
    await run_sidekick(
        task="Kendi staj notlarımdan MCP'nin ne işe yaradığını açıkla.",
        success_criteria=(
            "Cevap kullanıcının kendi notlarından bulunmalı ve "
            "MCP'nin agent ile dış sistemler arasındaki rolünü açıklamalı."
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
