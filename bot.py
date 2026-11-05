import telebot
import json
import requests
import whisper
import sqlite3
import os
import dspy
import html
from dspy.teleprompt import BootstrapFewShot

# ================= CONFIGURAÇÕES =================
API_TOKEN = os.environ["TELEGRAM_TOKEN"]
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
    """Gera uma consulta SQL SELECT a partir de uma pergunta em linguagem natural.
    Regras:
    -Sempre inclua a coluna 'produto' no SELECT, junto com as colunas pedidas. 
    -NÃO use SUM, COUNT ou outras agregações quando a pergunta for sobre um produto específico.
    - Para buscar um produto pelo nome, use LIKE com UMA ÚNICA palavra-chave, sem espaços nem hífens (ex: '%coca%', '%agua%'). Nunca use '='.
    - Escreva a palavra do produto sem acento e em minúsculas (ex: 'agua', não 'água').
    - Para quantidade: SELECT produto, quantidade FROM estoque WHERE produto LIKE '%...%'."""
    dbschema = dspy.InputField(desc="Schema do banco de dados")
    question = dspy.InputField(desc="Pergunta em linguagem natural")
    sql_query = dspy.OutputField(desc="Consulta SQL SELECT válida")

class ReliableSQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_sql = dspy.ChainOfThought(TextToSQL)

    def forward(self, question,schema=DB_SCHEMA):
        return self.generate_sql(dbschema=schema, question=question)

    
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
        dspy.Example(question="Quantos sabonetes temos?", sql_query="SELECT produto, quantidade FROM estoque WHERE produto LIKE '%sabonete%';").with_inputs("question"),
        dspy.Example(question="Qual a validade da coca cola?", sql_query="SELECT produto, data_validade FROM estoque WHERE produto LIKE '%coca%';").with_inputs("question"),
        dspy.Example(question="Qual o departamento do detergente?", sql_query="SELECT departamento, produto FROM estoque WHERE produto LIKE '%detergente%';").with_inputs("question"),
        dspy.Example(question="Qual o departamento da coca cola?", sql_query="SELECT departamento, produto FROM estoque WHERE produto LIKE '%coca%';").with_inputs("question"),
        dspy.Example(question="Quantas águas temos?", sql_query="SELECT produto, quantidade FROM estoque WHERE produto LIKE '%agua%';").with_inputs("question"),
        dspy.Example(question="Quais produtos vencem antes de 2026-12-31?", sql_query="SELECT produto, data_validade FROM estoque WHERE data_validade < '2026-12-31';").with_inputs("question"),
        dspy.Example(question="Quais produtos ja venceram?", sql_query="SELECT produto, data_validade FROM estoque WHERE data_validade < date('now');").with_inputs("question"),
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
        pred = gerador(question=question)
        sql = validate_sql(pred.sql_query)
        sql = validate_sql(pred.sql_query)
        print("SQL gerado:", sql)

        response = requests.post(SERVER_URL, json={"sql": sql})
        if response.status_code != 200:
            raise Exception("Erro ao consultar a API.")
        
        dados = response.json()["data"]
        # uma linha -> objeto; várias linhas -> lista
        return dados[0] if len(dados) == 1 else dados
    except Exception as e:
        return {"erro": str(e)}

def enviar_json(message, result):
    texto = json.dumps(result, ensure_ascii=False, indent=2)[:3900]
    texto = html.escape(texto)  # evita problemas com <, > e &
    try:
        bot.reply_to(message, f"<pre>{texto}</pre>", parse_mode="HTML")
    except Exception:
        # fallback: manda como texto puro, sem formatação
        bot.reply_to(message, texto)

@bot.message_handler(content_types=["text"])
def reply_text(message):
    enviar_json(message, ask_database(message.text))


if __name__ == "__main__":
    # PERGUNTA AO USUÁRIO SE QUER OTIMIZAR ANTES DE RODAR
    escolha = input("Deseja rodar o otimizador do DSPy antes de iniciar o bot? (s/n): ")
    if escolha.lower() == 's':
        otimizar_modelo()
        
    print("Iniciando o Bot do Telegram...")
    bot.polling()