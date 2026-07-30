from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class EvaluationResult(BaseModel):
    success: bool = Field(
        description="Başarı kriterlerinin karşılanıp karşılanmadığı."
    )
    feedback: str = Field(
        description="Sonucun neden başarılı/başarısız olduğu ve gerekiyorsa neyin düzeltilmesi gerektiği."
    )


evaluator = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).with_structured_output(EvaluationResult)


async def evaluate_result(
    task: str,
    success_criteria: str,
    answer: str,
) -> EvaluationResult:

    prompt = f"""
Sen Sidekick sisteminin bağımsız değerlendiricisisin.

KULLANICI GÖREVİ:
{task}

BAŞARI KRİTERLERİ:
{success_criteria}

SIDEKICK'İN SON CEVABI:
{answer}

Sidekick'in yaptığı işi başarı kriterlerine göre değerlendir.

Kurallar:
- Sadece verilen başarı kriterlerine göre karar ver.
- Kriterler karşılanıyorsa success=true döndür.
- Karşılanmıyorsa success=false döndür.
- Başarısızsa feedback alanında neyin eksik olduğunu açıkça belirt.
"""

    return await evaluator.ainvoke(prompt)
