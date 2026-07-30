from langchain.agents.middleware import (
    TodoListMiddleware,
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    HumanInTheLoopMiddleware,
)

from rag.rag_tool import search_my_notes

"""
Faz 4 — MCP (GitHub) + Email (SMTP/IMAP)
create_agent() + Checkpointer + @tool'lar + GitHub MCP tool'ları.
MCP bağlantısı async olduğu için agent kurulumu da async.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.basic_tools import get_current_datetime, save_note
from tools.email_tool import send_email, read_recent_emails
from mcp_servers.github_mcp import get_github_tools
from mcp_servers.playwright_mcp import get_playwright_tools
from mcp_servers.calendar_mcp import get_calendar_tools

load_dotenv()


SYSTEM_PROMPT = """
Sen kullanıcının kişisel AI asistanısın, adın Sidekick.

Kişiliğin:
- Samimi ve doğal konuş, resmi bir chatbot gibi davranma.
- Kullanıcıyla kişisel bir arkadaş/iş arkadaşı gibi iletişim kur.
- Her zaman doğal, akıcı ve dilbilgisel olarak doğru Türkçe kullan.
- "Seni nasıl yardımcı olabilirim?" gibi bozuk ifadeler kullanma.
- Doğru kullanım: "Sana nasıl yardımcı olabilirim?"
- Cümlelerini göndermeden önce Türkçe dilbilgisi açısından kontrol et.
- Basit sohbetlerde gereksiz araç kullanma, sadece doğal cevap ver.
- Görev gerektiğinde uygun aracı kullan: tarih/saat, not kaydetme,
  mail okuma/gönderme veya GitHub repo bilgisi.
- Kısa ve net konuş, gereksiz uzatma.
- Önceki mesajları hatırlıyorsun, bağlamı koru.

Todo kullanımı:
- Kullanıcı birden fazla adım içeren görev verdiğinde,
  herhangi bir başka tool kullanmadan ÖNCE mutlaka write_todos aracını çağır.
- Çok adımlı görevlerde write_todos kullanmak zorunludur.
- Her ana görev için yalnızca bir todo maddesi oluştur.
- Todo maddelerini kısa tut; tercihen 3-8 kelime.
- Araştırma sonuçlarını, GitHub repo isimlerini, linkleri,
  açıklamaları veya uzun metinleri todo içine YAZMA.
- Todo listesi sadece yapılacak ana adımları göstermeli.
- Bir adım tamamlandığında yalnızca status bilgisini güncelle.
- Detaylı sonuçları normal cevabında ver.

E-posta kuralları:
- Kullanıcı "kendi mail adresime", "kendime mail gönder",
  "benim mailime gönder" derse hedef e-posta adresi olarak
  SMTP_EMAIL ortam değişkenindeki adresi kullan.
- Kullanıcı açıkça başka bir e-posta adresi verirse,
  o adresi olduğu gibi kullan.
- E-posta adreslerini [REDACTED_EMAIL] gibi değiştirme.
- E-posta gönderirken `to` alanına kullanıcı adı değil,
  tam e-posta adresi yazılmalıdır.
- send_email aracı kullanıcı onayı olmadan çalıştırılmamalıdır.

GitHub araç kullanım kuralları:
- Kullanıcının kendi GitHub kullanıcı adı: byznrdincer.
- Kullanıcı "GitHub hesabım", "repolarımı", "repositorylerimi",
  "projelerimi" gibi kendi hesabından bahsediyorsa
  GitHub kullanıcısı olarak byznrdincer kullan.
- Kullanıcının repolarını listelemek için
  search_repositories aracını query="user:byznrdincer" ile çağır.
- Bu sorgudan dönen items listesini repository listesi olarak kullanıcıya sun.
- Sonuç boş değilse asla "repository yok" deme.

- Kullanıcı GitHub hesabındaki repolarını/repository'lerini isterse
  search_repositories aracını kullan.
- Kullanıcı adı gerekiyorsa önce get_me aracını kullan.
- Repository listesini almak için list_releases,
  get_latest_release veya release araçlarını kullanma.
- list_releases yalnızca kullanıcı özellikle bir repository'nin
  release'lerini istediğinde kullanılmalıdır.
- Bir GitHub aracı hata verirse bunun kullanıcının hiç repository'si
  olmadığı anlamına geldiğini varsayma; uygun başka GitHub aracını dene.
"""


def handle_tool_error(error, request):
    return (
        f"Tool çalıştırılırken hata oluştu: {error}. "
        "Bu aracın başarısız olduğunu kullanıcıya açıkla ve "
        "mümkünse başka bir araç veya yöntem kullan."
    )


checkpointer = InMemorySaver()


async def build_agent():
    """Tool + MCP + memory'li Worker Agent'ı async olarak kurar."""

    github_tools = await get_github_tools()
    playwright_tools = await get_playwright_tools()

    all_tools = [
        get_current_datetime,
        save_note,
        send_email,
        read_recent_emails,
        search_my_notes,
    ] + github_tools + playwright_tools

    my_email = os.getenv("SMTP_EMAIL", "")

    runtime_prompt = SYSTEM_PROMPT + f"""

Kullanıcının kendi e-posta adresi: {my_email}

Kullanıcı kendine mail göndermeni isterse
`to` alanında tam olarak bu adresi kullan.

Kullanıcı mesajında açıkça farklı bir e-posta adresi verirse,
o adresi değiştirmeden kullan.
"""

    agent = create_agent(
        model="gpt-4o-mini",
        tools=all_tools,
        system_prompt=runtime_prompt,
        middleware=[
            TodoListMiddleware(),

            ModelCallLimitMiddleware(
                run_limit=30
            ),

            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,
                }
            ),

            ToolErrorMiddleware(
                on_error=handle_tool_error
            ),
        ],
        checkpointer=checkpointer,
    )

    return agent


async def _test():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et."
        )

    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        raise RuntimeError(
            "GITHUB_PERSONAL_ACCESS_TOKEN bulunamadı. .env dosyanı kontrol et."
        )

    agent = await build_agent()

    config = {
        "configurable": {
            "thread_id": "test-thread-email"
        }
    }

    # Test 1: Mail okuma
    r1 = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Gelen kutumdaki son 3 maili göster."
                }
            ]
        },
        config=config,
    )

    print(
        "Sidekick (okuma testi):",
        r1["messages"][-1].content,
    )

    print("\n" + "=" * 50 + "\n")

    # Test 2: Mail gönderme
    my_email = os.getenv("SMTP_EMAIL")

    r2 = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{my_email} adresine, "
                        "konusu 'Sidekick Test' olan, "
                        "içeriği 'Bu bir test mailidir, "
                        "Faz 4 email entegrasyonu çalışıyor.' "
                        "olan bir mail gönder."
                    ),
                }
            ]
        },
        config=config,
    )

    print(
        "Sidekick (gönderme testi):",
        r2["messages"][-1].content,
    )

    print("\n" + "=" * 50 + "\n")

    # Test 3: Playwright ile web araştırması
    r3 = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "https://example.com sitesine git "
                        "ve sayfadaki başlığı söyle."
                    ),
                }
            ]
        },
        config=config,
    )

    print(
        "Sidekick (browser testi):",
        r3["messages"][-1].content,
    )


if __name__ == "__main__":
    asyncio.run(_test())