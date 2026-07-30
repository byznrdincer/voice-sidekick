"""
Faz 4 — Kendi Email Tool'larımız.
send_email: SMTP ile mail gönderir (kritik işlem, Human-in-the-Loop Faz 7'de eklenecek).
read_recent_emails: IMAP ile son gelen mailleri okur.
"""

import os
import imaplib
import email as email_lib
import smtplib
from email.mime.text import MIMEText
from email.header import decode_header
from langchain.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Verilen adrese SMTP üzerinden mail gönderir.
    to: alıcı email adresi
    subject: mail konusu
    body: mail içeriği (düz metin)
    Bu kritik bir işlemdir, kullanıcı onayı olmadan asla çağırma."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_APP_PASSWORD")

    if not all([host, user, password]):
        return "Hata: SMTP bilgileri .env dosyasında eksik."

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to], msg.as_string())
        return f"Mail başarıyla gönderildi: {to}"
    except Exception as e:
        return f"Mail gönderilemedi: {e}"


def _decode(value: str) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    decoded = ""
    for text, enc in parts:
        if isinstance(text, bytes):
            decoded += text.decode(enc or "utf-8", errors="ignore")
        else:
            decoded += text
    return decoded


@tool
def read_recent_emails(count: int = 5) -> str:
    """Gelen kutusundaki en son mailleri okur (IMAP).
    count: kaç mail getirileceği (varsayılan 5).
    Kullanıcı 'bugünkü maillerim', 'gelen kutum', 'son mailler' gibi
    bir şey sorduğunda bu aracı kullan."""
    host = os.getenv("IMAP_HOST")
    port = int(os.getenv("IMAP_PORT", "993"))
    user = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_APP_PASSWORD")

    if not all([host, user, password]):
        return "Hata: IMAP bilgileri .env dosyasında eksik."

    try:
        mail = imaplib.IMAP4_SSL(host, port)
        mail.login(user, password)
        mail.select("inbox")

        status, data = mail.search(None, "ALL")
        ids = data[0].split()
        latest_ids = ids[-count:] if len(ids) > count else ids
        latest_ids.reverse()

        results = []
        for mid in latest_ids:
            status, msg_data = mail.fetch(mid, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw_email)
            subject = _decode(msg.get("Subject", ""))
            sender = _decode(msg.get("From", ""))
            date = msg.get("Date", "")
            results.append(f"- Kimden: {sender}\n  Konu: {subject}\n  Tarih: {date}")

        mail.logout()

        if not results:
            return "Gelen kutusunda mail bulunamadı."
        return "\n".join(results)
    except Exception as e:
        return f"Mailler okunamadı: {e}"
