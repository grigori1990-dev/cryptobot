"""
КриптоБот — ОСТАНОВЛЕН
Идёт пересборка с нуля на основе бэктеста.
"""
import os
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

if TELEGRAM_TOKEN and CHAT_ID:
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": "🔴 <b>КриптоБот остановлен.</b>\nИдёт пересборка с нуля на основе бэктеста. Скоро вернёмся.",
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram error: {e}", flush=True)

print("Бот остановлен. Ожидаем пересборки.", flush=True)
