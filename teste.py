import os, time, dspy
os.environ.setdefault("TELEGRAM_TOKEN", "123:abc")
import bot

for i in range(2):
    t = time.time()
    try:
        pred = bot.ReliableSQLGenerator()(question="Quantas águas temos?")
        print("SQL:", pred.sql_query)
    except Exception as e:
        print("ERRO:", e)
    print(f"tentativa {i+1}: {time.time() - t:.1f}s")