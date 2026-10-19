import telebot
import json
import requests
import whisper
import sqlite3
import os
import dspy
from dspy.teleprompt import BootstrapFewShot

# ================= CONFIGURAÇÕES =================
API_TOKEN = "TELEGRAM_TOKEN"
SERVER_URL = "http://localhost:8000/query"
ARQUIVO_OTIMIZADO = "gerador_otimizado.json"

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

# DSPy Configuração
lm = dspy.LM("openai/gemma-4-E2B-it-IQ4_XS", api_base="http://localhost:1337/v1", api_key="not-needed")
dspy.configure(lm=lm)

bot = telebot.TeleBot(API_TOKEN)

# ================= MÓDULOS DE IA =================
class TextToSQL(dspy.Signature):
    """Gera uma consulta SQL SELECT a partir de uma pergunta em linguagem natural."""
    dbschema = dspy.InputField(desc="Schema do banco de dados")
    question = dspy.InputField(desc="Pergunta em linguagem natural")
    sql_query = dspy.OutputField(desc="Consulta SQL SELECT válida")

class ReliableSQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_sql = dspy.ChainOfThought(TextToSQL)

    def forward(self, schema, question):
        return self.generate_sql(dbschema=schema, question=question)
class GenerateResponse(dspy.Signature):
        """Gera uma resposta amigável em linguagem natural baseada nos dados do banco."""
        question = dspy.InputField(desc="Pergunta original do usuário")
        data = dspy.InputField(desc="Dados brutos retornados do banco em JSON")
        answer = dspy.OutputField(desc="Resposta em linguagem natural, curta e direta")

# ================= VALIDAÇÃO (Requisitos 2 e 3) =================
def validate_sql(sql):
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    
    if not sql.upper().startswith("SELECT"):
        raise ValueError("Erro de segurança: O SQL gerado não é um SELECT.")

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(DB_SCHEMA)
        conn.execute(sql)
    except sqlite3.Error as e:
        raise ValueError(f"SQL inválido gerado pela IA: {e}")
    finally:
        conn.close()
    
    return sql

# ================= OTIMIZADOR (Requisito 4) =================
def metric_validacao(example, pred, trace=None):
    """Métrica para o otimizador: o SQL deve ser válido e fazer match com o esperado."""
    try:
        sql_valido = validate_sql(pred.sql_query)
        return example.sql_query.strip().lower() in sql_valido.lower()
    except Exception:
        return False

def otimizar_modelo():
    """Roda o otimizador do DSPy e salva o modelo."""
    print("Iniciando otimização do Prompt com GEPA/BootstrapFewShot...")
    dataset = [
        dspy.Example(question="Quantos sabonetes temos?", sql_query="SELECT quantidade FROM estoque WHERE produto = 'sabonete';").with_inputs("question"),
        dspy.Example(question="Qual a validade da coca-cola?", sql_query="SELECT data_validade FROM estoque WHERE produto = 'coca-cola lata';").with_inputs("question")
    ]
    
    teleprompter = BootstrapFewShot(metric=metric_validacao, max_bootstrapped_demos=2, max_labeled_demos=2)
    gerador = ReliableSQLGenerator()
    gerador_otimizado = teleprompter.compile(gerador, trainset=dataset)
    gerador_otimizado.save(ARQUIVO_OTIMIZADO)
    print(f"Otimização salva em {ARQUIVO_OTIMIZADO}!")

# ================= INTEGRAÇÃO E TELEGRAM =================
def ask_database(question):
    gerador = ReliableSQLGenerator()
    
    if os.path.exists(ARQUIVO_OTIMIZADO):
        gerador.load(ARQUIVO_OTIMIZADO)
        
    try:
        pred = gerador(schema=DB_SCHEMA, question=question)
        sql = validate_sql(pred.sql_query)
        print(f"SQL Gerado e Validado: {sql}")

        response = requests.post(SERVER_URL, json={"sql": sql})
        if response.status_code != 200:
            raise Exception("Erro ao consultar a API.")
        
        dados_brutos = response.json()["data"]
        
        tradutor = dspy.Predict(GenerateResponse)
        resposta_ia = tradutor(question=question, data=str(dados_brutos))
        
        return resposta_ia.answer
        
    except Exception as e:
        print(f"Erro no processo: {e}")
        raise

@bot.message_handler(content_types=["text"])
def reply_text(message):
    try:
        result = ask_database(message.text)
        bot.reply_to(message, result)
    except Exception as e:
        bot.reply_to(message, f"Erro: {e}")

def whisper_transcribe(filepath: str, model="tiny") -> str:
    modelo = whisper.load_model(model)
    result = modelo.transcribe(filepath)
    return result["text"]

@bot.message_handler(content_types=["voice"])
def transcribe_voice_message(message):
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        filepath = "audio.ogg"
        
        with open(filepath, "wb") as f:
            f.write(downloaded_file)
            
        text = whisper_transcribe(filepath)
        result = ask_database(text)
        bot.reply_to(message, json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        bot.reply_to(message, f"Erro: {e}")

if __name__ == "__main__":
    # PERGUNTA AO USUÁRIO SE QUER OTIMIZAR ANTES DE RODAR
    escolha = input("Deseja rodar o otimizador do DSPy antes de iniciar o bot? (s/n): ")
    if escolha.lower() == 's':
        otimizar_modelo()
        
    print("Iniciando o Bot do Telegram...")
    bot.polling()