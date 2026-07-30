"""
Faz 3 — Kendi @tool'larımız.
İlk aşamada iki basit tool: tarih/saat öğrenme ve dosyaya not yazma.
"""

import datetime
from langchain.tools import tool


@tool
def get_current_datetime() -> str:
    """Şu anki tarih ve saati döner. Kullanıcı 'bugün günlerden ne',
    'saat kaç', 'tarih ne' gibi sorular sorduğunda bu aracı kullan."""
    now = datetime.datetime.now()
    return now.strftime("%d %B %Y, %A — %H:%M")


@tool
def save_note(filename: str, content: str) -> str:
    """Kullanıcının istediği bir notu/metni dosyaya kaydeder.
    filename: dosya adı (örn. 'not.md', 'ozet.txt')
    content: dosyaya yazılacak içerik.
    Kullanıcı 'bunu bir dosyaya kaydet', 'not al', 'yaz bunu' dediğinde kullan."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"'{filename}' dosyasına kaydedildi ({len(content)} karakter)."
