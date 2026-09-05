import json
import sqlite3
import requests

import dspy
import telebot
import whisper


API_TOKEN = ""

SERVER_URL = "http://localhost:8000/query"

# DSPy
lm = dspy.LM("openai/gemma-4-E2B-it-IQ4_XS",api_base="http://localhost:1337/v1",api_key="not-needed")

dspy.configure(lm=lm)

# SCHEMA DO BANCO
DB_SCHEMA = """
CREATE TABLE estoque (
    id INTEGER,
    produto TEXT,
    departamento TEXT,
    data_fabricacao TEXT,
    data_validade TEXT,
    quantidade INTEGER
);
"""

# GERADOR DE SQL
class TextToSQL(dspy.Signature):
    """
    Gera uma consulta SQL SELECT a partir de uma pergunta
    em linguagem natural.

    A consulta deve utilizar somente a tabela estoque.
    """

    dbschema = dspy.InputField(
        desc="Schema do banco de dados"
    )

    question = dspy.InputField(
        desc="Pergunta em linguagem natural"
    )

    sql_query = dspy.OutputField(
        desc="Consulta SQL SELECT válida"
    )


class ReliableSQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_sql = dspy.ChainOfThought(
            TextToSQL
        )

    def forward(self, schema, question):
        pred = self.generate_sql(
            dbschema=schema,
            question=question
        )
        return pred

# VALIDAÇÃO DO SQL
def validate_sql(sql):
    sql = sql.strip()

    # Remove possíveis blocos de código gerados pela IA
    if sql.startswith("```"):
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

    # somente SELECT pode ser executado
    if not sql.upper().startswith("SELECT"):
        raise ValueError(
            "O SQL gerado não é um SELECT."
        )

    # testa o SQL em SQLite em memória
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("""
            CREATE TABLE estoque (
                id INTEGER,
                produto TEXT,
                departamento TEXT,
                data_fabricacao TEXT,
                data_validade TEXT,
                quantidade INTEGER
            )
        """)
        conn.execute(sql)

    except sqlite3.Error as e:
        raise ValueError(
            f"SQL inválido: {e}"
        )

    finally:
        conn.close()

    return sql

# GERAR SQL
def generate(question):
    generator = ReliableSQLGenerator()
    try:
        prediction = generator.forward(
            DB_SCHEMA,
            question
        )

        sql = prediction.sql_query

        sql = validate_sql(sql)

        return sql

    except Exception as e:
        print(f"Erro ao gerar/validar SQL: {e}")

        raise

# EXECUTAR NO SERVIDOR
def execute_sql(sql):
    response = requests.post(
        SERVER_URL,
        json={
            "sql": sql
        }
    )

    if response.status_code != 200:
        raise Exception(
            response.json().get(
                "detail",
                "Erro ao consultar o servidor."
            )
        )
    return response.json()["data"]

# CONSULTA COMPLETA
def ask_database(question):
    sql = generate(question)

    print("SQL gerado:")
    print(sql)

    result = execute_sql(sql)

    return result

# TELEGRAM
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(content_types=["text"])
def reply_text(message):
    try:
        result = ask_database(
            message.text
        )

        bot.reply_to(
            message,
            json.dumps(
                result,
                ensure_ascii=False
            )
        )

    except Exception as e:
        bot.reply_to(message,f"Erro: {e}")

# WHISPER
def whisper_transcribe(
    filepath: str,
    model="tiny"
) -> str:
    model = whisper.load_model(model)

    result = model.transcribe(filepath)

    return result["text"]

@bot.message_handler(
    content_types=["voice"]
)
def transcribe_voice_message(message):
    try:
        file_info = bot.get_file(
            message.voice.file_id
        )

        downloaded_file = bot.download_file(
            file_info.file_path
        )
        filepath = "audio.ogg"

        with open(filepath, "wb") as f:
            f.write(downloaded_file)

        text = whisper_transcribe(
            filepath
        )
        result = ask_database(text)

        bot.reply_to(
            message,
            json.dumps(
                result,
                ensure_ascii=False
            )
        )

    except Exception as e:
        bot.reply_to(
            message,
            f"Erro: {e}"
        )

# MAIN
if __name__ == "__main__":

    print("Bot iniciado...")

    bot.polling()