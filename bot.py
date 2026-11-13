import telebot
import json
import requests
import sqlite3
import os
import time
import dspy
import html
from dspy.teleprompt import BootstrapFewShot

# [NOVO] dados do estoque, usados só pela métrica do GEPA (importar não sobe a API)
from server import SEED_DATA

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
# [ALTERADO] temperature=0 e max_tokens=200: resposta determinística e curta (mais rápido)
lm = dspy.LM("openai/gemma-4-E2B-it-IQ4_XS", model_type="chat",
             api_base="http://localhost:1337/v1", api_key="not-needed",
             temperature=0, max_tokens=200,
             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
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
        # [ALTERADO] Predict no lugar de ChainOfThought: o modelo não escreve raciocínio
        # antes do SQL, então gera bem menos tokens. (Para voltar: dspy.ChainOfThought)
        self.generate_sql = dspy.Predict(TextToSQL)

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

# [NOVO] métrica do GEPA: compara o RESULTADO da consulta (não o texto do SQL) e devolve
# um feedback em texto, que o GEPA usa para reescrever as instruções do prompt.
def _executar(sql):
    conn = sqlite3.connect(":memory:")
    conn.execute(DB_SCHEMA)
    conn.executemany(
        "INSERT INTO estoque (produto, departamento, data_fabricacao, data_validade, quantidade) VALUES (?, ?, ?, ?, ?)",
        SEED_DATA,
    )
    try:
        linhas = conn.execute(sql).fetchall()
    finally:
        conn.close()
    # ignora a ordem das colunas e das linhas
    return sorted(tuple(sorted(str(v) for v in linha)) for linha in linhas)

def metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        obtido = _executar(validate_sql(pred.sql_query))
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"O SQL é inválido ou inseguro: {e}. Gere apenas um SELECT válido para a tabela estoque.")
    esperado = _executar(gold.sql_query)
    if obtido == esperado:
        return dspy.Prediction(score=1.0, feedback="Correto.")
    if not obtido:
        return dspy.Prediction(score=0.0, feedback="A consulta retornou 0 linhas: a palavra do LIKE deve estar errada (acento, hífen, plural ou mais de uma palavra) ou o filtro está incorreto.")
    return dspy.Prediction(score=0.3, feedback=f"Resultado diferente do esperado. Esperado: {esperado}. Obtido: {obtido}. Revise as colunas do SELECT e o filtro WHERE.")

# [ALTERADO] ganhou o parâmetro "modo" ("bootstrap" ou "gepa"); o dataset continua o mesmo
def otimizar_modelo(modo="bootstrap"):
    """Roda o otimizador do DSPy e salva o modelo."""
    print(f"Iniciando otimização do Prompt com {modo}...")
    dataset = [
        dspy.Example(question="Quantos sabonetes temos?", sql_query="SELECT produto, quantidade FROM estoque WHERE produto LIKE '%sabonete%';").with_inputs("question"),
        dspy.Example(question="Qual a validade da coca cola?", sql_query="SELECT produto, data_validade FROM estoque WHERE produto LIKE '%coca%';").with_inputs("question"),
        dspy.Example(question="Qual o departamento do detergente?", sql_query="SELECT departamento, produto FROM estoque WHERE produto LIKE '%detergente%';").with_inputs("question"),
        dspy.Example(question="Qual o departamento da coca cola?", sql_query="SELECT departamento, produto FROM estoque WHERE produto LIKE '%coca%';").with_inputs("question"),
        dspy.Example(question="Quantas águas temos?", sql_query="SELECT produto, quantidade FROM estoque WHERE produto LIKE '%agua%';").with_inputs("question"),
        dspy.Example(question="Quais produtos vencem antes de 2026-12-31?", sql_query="SELECT produto, data_validade FROM estoque WHERE data_validade < '2026-12-31';").with_inputs("question"),
        dspy.Example(question="Quais produtos ja venceram?", sql_query="SELECT produto, data_validade FROM estoque WHERE data_validade < date('now');").with_inputs("question"),
    ]
    
    gerador = ReliableSQLGenerator()
    if modo == "gepa":
        reflection_lm = dspy.LM("openai/gemma-4-E2B-it-IQ4_XS", model_type="chat",
                                api_base="http://localhost:1337/v1", api_key="not-needed",
                                temperature=1.0, max_tokens=1000,
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    else:
        teleprompter = BootstrapFewShot(metric=metric_validacao, max_bootstrapped_demos=2, max_labeled_demos=2)
        gerador_otimizado = teleprompter.compile(gerador, trainset=dataset)
    gerador_otimizado.save(ARQUIVO_OTIMIZADO)
    print(f"Otimização salva em {ARQUIVO_OTIMIZADO}!")

# ================= INTEGRAÇÃO E TELEGRAM =================
# [NOVO] o gerador é criado e o .json lido UMA vez (antes era a cada pergunta)
_gerador = None

def carregar_gerador():
    global _gerador
    if _gerador is None:
        _gerador = ReliableSQLGenerator()
        if os.path.exists(ARQUIVO_OTIMIZADO):
            try:
                _gerador.load(ARQUIVO_OTIMIZADO)
            except Exception as e:
                # .json salvo com outro formato (ex.: gerado com ChainOfThought)
                print(f"Não consegui carregar {ARQUIVO_OTIMIZADO} ({e}). Usando o prompt original; apague o arquivo e otimize de novo.")
                _gerador = ReliableSQLGenerator()
    return _gerador

def ask_database(question):
    try:
        t0 = time.time()
        pred = carregar_gerador()(question=question)
        # [ALTERADO] removida a linha duplicada de validate_sql
        sql = validate_sql(pred.sql_query)
        print(f"SQL gerado: {sql} (LLM: {time.time() - t0:.1f}s)")

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
    bot.send_chat_action(message.chat.id, "typing")  # [NOVO] mostra "digitando..." na hora
    enviar_json(message, ask_database(message.text))


if __name__ == "__main__":
    # [ALTERADO] agora tem 3 opções: não otimizar / Bootstrap / GEPA
    escolha = input("Otimizar o prompt antes de iniciar o bot? (n = não / b = Bootstrap / g = GEPA): ").lower()
    if escolha == 'g':
        otimizar_modelo("gepa")
    elif escolha in ('b', 's'):
        otimizar_modelo("bootstrap")

    carregar_gerador()  # [NOVO] deixa o gerador pronto antes da primeira pergunta
    print("Iniciando o Bot do Telegram...")
    bot.polling()