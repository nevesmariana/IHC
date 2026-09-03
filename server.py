"""
server.py - Passo 1: encapsular o BD e expor via FastAPI com uma rota GET.
Por enquanto: cria o banco de estoque e devolve os dados via HTTP.
Próximo passo: requisito 2 -> 
"""
import os
import sqlite3
from fastapi  import FastAPI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "estoque.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS estoque (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    produto TESXT NOT NULL,
    departamento TEXT NOT NULL,
    data_fabricacao TEXT,
    data_validade TEXT,
    quantidade INTEGER DEFAULT 0)
"""

SEED_DATA = [
    ("sabonete", "higiene", "2025-01-10", "2027-01-10", 120),
    ("agua mineral 500 ml", "bebidas", "2025-06-01", "2026-06-01", 500),
    ("coca-cola lata", "bebidas", "2025-05-15", "2026-05-15", 300),
    ("detergente", "limpeza", "2025-03-01", "2027-03-01", 200),
]
#seed_data, contem dados de exemplos, só para consultar.
#cada tupla vira uma linha na tebala estoque 


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA_SQL)

    count = conn.execute("SELECT COUNT(*) FROM estoque").fetchone()[0]
    if count ==0:
        conn.executemany(
            "INSERT INTO estoque "
            "(produto, departamento, data_fabricacao, data_validade, quantidade) "
            "VALUES (?, ?, ?, ?, ?)",
            SEED_DATA
        )
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Estoque API")

@app.get("/estoque")
def get_estoque():
    """Retorna todos os itens do etoque. """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM  estoque").fetchall()
    conn.close()

    return {"data": [dict(row) for row in rows]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)