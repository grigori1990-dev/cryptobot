#!/usr/bin/env python3
"""CryptoBot System Audit — полный технический отчёт"""

import os
import glob
import re
from datetime import datetime

REPORT = []

def section(title):
    REPORT.append("\n" + "="*60)
    REPORT.append(f"  {title}")
    REPORT.append("="*60)

def add(text):
    REPORT.append(str(text))

# ─────────────────────────────────────────────
# 1. КОНФИГУРАЦИЯ ЗАПУСКА
# ─────────────────────────────────────────────
section("1. КОНФИГУРАЦИЯ ЗАПУСКА")

base = os.path.dirname(os.path.abspath(__file__))

# GitHub Actions workflows
wf_dir = os.path.join(base, ".github", "workflows")
if os.path.isdir(wf_dir):
    wf_files = glob.glob(os.path.join(wf_dir, "*.yml")) + glob.glob(os.path.join(wf_dir, "*.yaml"))
    if wf_files:
        for f in wf_files:
            add(f"\n📄 {os.path.relpath(f, base)}")
            add("-"*40)
            with open(f) as fh:
                add(fh.read())
    else:
        add("⚠️  Папка .github/workflows/ есть, но yml файлов нет")
else:
    add("⚠️  Папка .github/workflows/ НЕ НАЙДЕНА")

# Dockerfile / docker-compose / Procfile
for fname in ["Dockerfile", "docker-compose.yml", "docker-compose.yaml", "Procfile", ".env.example"]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        add(f"\n📄 {fname}")
        add("-"*40)
        with open(fpath) as fh:
            add(fh.read())
    else:
        add(f"○  {fname} — не найден")

# ─────────────────────────────────────────────
# 2. ЛОГИКА ВРЕМЕНИ И ПЕРИОДИЧНОСТИ
# ─────────────────────────────────────────────
section("2. ЛОГИКА ВРЕМЕНИ И ПЕРИОДИЧНОСТИ")

main_candidates = ["main.py", "bot.py", "app.py", "run.py"]
main_file = None
for c in main_candidates:
    p = os.path.join(base, c)
    if os.path.exists(p):
        main_file = p
        break

if main_file:
    add(f"\n📄 Анализируется: {os.path.basename(main_file)}\n")
    with open(main_file) as fh:
        lines = fh.readlines()

    patterns = [
        ("schedule", r"schedule\."),
        ("time.sleep", r"time\.sleep"),
        ("while True", r"while\s+True"),
        ("cron", r"cron"),
        ("TRADE_START/END", r"TRADE_START|TRADE_END|trading_hours|is_trading"),
        ("timeout", r"timeout"),
        ("PYTHONUNBUFFERED", r"PYTHONUNBUFFERED"),
    ]

    for label, pattern in patterns:
        matches = [(i+1, line.rstrip()) for i, line in enumerate(lines) if re.search(pattern, line, re.IGNORECASE)]
        if matches:
            add(f"\n🔍 [{label}] — найдено {len(matches)} вхождений:")
            for lineno, text in matches:
                add(f"   Строка {lineno:4d}: {text}")

    # Константы времени
    add("\n\n🕐 Константы времени из кода:")
    time_consts = [(i+1, l.rstrip()) for i, l in enumerate(lines)
                   if re.search(r"(TRADE_START|TRADE_END|MIN_FACTORS|TOTAL_FACTORS|timeout)\s*=", l)]
    for lineno, text in time_consts:
        add(f"   Строка {lineno:4d}: {text.strip()}")
else:
    add("❌ main.py / bot.py не найден")

# ─────────────────────────────────────────────
# 3. ЛОГИ ВЫПОЛНЕНИЯ
# ─────────────────────────────────────────────
section("3. ЛОГИ ВЫПОЛНЕНИЯ (последние 20 строк)")

log_candidates = [
    os.path.join(base, "last_run.log"),
    os.path.join(base, "latest_log.txt"),
    os.path.join(base, "bot.log"),
    "/tmp/bot.log",
]

found_log = False
for lf in log_candidates:
    if os.path.exists(lf) and os.path.getsize(lf) > 0:
        add(f"\n📋 Файл лога: {lf} ({os.path.getsize(lf)} байт)")
        add(f"   Изменён: {datetime.fromtimestamp(os.path.getmtime(lf)).strftime('%Y-%m-%d %H:%M:%S')}")
        add("-"*40)
        with open(lf) as fh:
            log_lines = fh.readlines()
        last20 = log_lines[-20:] if len(log_lines) >= 20 else log_lines
        for line in last20:
            add(line.rstrip())
        found_log = True
        break

if not found_log:
    add("⚠️  Файлы логов не найдены или пусты.")
    add("   Логи хранятся на серверах GitHub Actions — скачивай через check_actions.command")

# ─────────────────────────────────────────────
# 4. ЗАВИСИМОСТИ
# ─────────────────────────────────────────────
section("4. ЗАВИСИМОСТИ (requirements.txt)")

req_path = os.path.join(base, "requirements.txt")
if os.path.exists(req_path):
    with open(req_path) as fh:
        content = fh.read().strip()
    add(f"\n📦 Найдено в requirements.txt:\n")
    for line in content.splitlines():
        if line.strip() and not line.startswith("#"):
            add(f"   ✓ {line.strip()}")
else:
    add("⚠️  requirements.txt не найден")
    add("   Зависимости устанавливаются в workflow: pip install ccxt pandas requests schedule pytz")

# ─────────────────────────────────────────────
# 5. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ И ТАЙМАУТЫ
# ─────────────────────────────────────────────
section("5. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ И ТАЙМАУТЫ")

env_vars = {
    "TELEGRAM_TOKEN": os.environ.get("TELEGRAM_TOKEN"),
    "CHAT_ID": os.environ.get("CHAT_ID"),
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED"),
    "GITHUB_ACTIONS": os.environ.get("GITHUB_ACTIONS"),
    "RUNNER_OS": os.environ.get("RUNNER_OS"),
    "GITHUB_RUN_ID": os.environ.get("GITHUB_RUN_ID"),
}

add("\n🔐 Переменные окружения:")
for k, v in env_vars.items():
    if v:
        masked = "***" if k in ("TELEGRAM_TOKEN", "CHAT_ID") else v
        add(f"   ✓ {k} = {masked}")
    else:
        add(f"   ○ {k} = не задана (используется значение из кода)")

# Таймаут из workflow
add("\n⏱  Настройки таймаутов:")
if main_file:
    wf_files_all = glob.glob(os.path.join(base, ".github", "workflows", "*.yml"))
    for wf in wf_files_all:
        with open(wf) as fh:
            wf_content = fh.read()
        timeouts = re.findall(r"timeout[^\n]*", wf_content, re.IGNORECASE)
        for t in timeouts:
            add(f"   Workflow: {t.strip()}")
        crons = re.findall(r"cron:\s*['\"]([^'\"]+)['\"]", wf_content)
        for c in crons:
            add(f"   Cron расписание: '{c}'")
            # Интерпретация
            parts = c.strip().split()
            if parts[0].startswith("*/"):
                add(f"   → Запуск каждые {parts[0][2:]} минут")
            elif parts[1].startswith("*/"):
                add(f"   → Запуск каждые {parts[1][2:]} часа/ов")

# ─────────────────────────────────────────────
# ИТОГ
# ─────────────────────────────────────────────
section("ИТОГ — ДИАГНОСТИКА")

add("""
  🤖 Бот: CryptoBot v3.0
  📡 Биржа: Bybit (переключён с Binance — US IP блокировка)
  ⏰ Расписание: GitHub Actions cron (каждые 10 мин по расписанию)
     ⚠️  GitHub НЕ гарантирует точное время — задержка 5-15 мин нормальна
  🔍 Логика скана: каждый запуск GitHub Actions = 1 скан рынка
  ⏱  Таймаут одного запуска: 540 секунд (9 минут)
  📊 Порог сигнала: MIN_FACTORS=6 из 13 (снижен для большего числа сигналов)
  🚫 Фильтр BTC тренда: ОТКЛЮЧЁН (раньше блокировал все сигналы)
  📱 Telegram: токен вшит напрямую в код (не зависит от GitHub Secrets)
""")

# ─────────────────────────────────────────────
# ВЫВОД И СОХРАНЕНИЕ
# ─────────────────────────────────────────────
report_text = "\n".join(REPORT)
report_path = os.path.join(base, "system_audit_report.txt")
with open(report_path, "w") as fh:
    fh.write(report_text)

print(report_text)
print(f"\n✅ Отчёт сохранён: {report_path}")
