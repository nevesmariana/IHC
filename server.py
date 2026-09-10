import sqlite3
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

DB_PATH = "estoque.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS estoque (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    produto TEXT NOT NULL,
    departamento TEXT NOT NULL,
    data_fabricacao TEXT,
    data_validade TEXT,
    quantidade INTEGER DEFAULT 0
)
"""

SEED_DATA = [
    ("sabonete", "higiene", "2025-01-10", "2027-01-10", 120),
    ("agua mineral 500 ml", "bebidas", "2025-06-01", "2026-06-01", 500),
    ("coca-cola lata", "bebidas", "2025-05-15", "2026-05-15", 300),
    ("detergente", "limpeza", "2025-03-01", "2027-03-01", 200),
]

def init_db():
    """Inicializa e popula o banco de dados caso esteja vazio."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA_SQL)
    count = conn.execute("SELECT COUNT(*) FROM estoque").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO estoque (produto, departamento, data_fabricacao, data_validade, quantidade) VALUES (?, ?, ?, ?, ?)",
            SEED_DATA
        )
    conn.commit()
    conn.close()

# Inicializa o banco de dados ao iniciar o arquivo
init_db()

app = FastAPI(title="Estoque API")

class SQLRequest(BaseModel):
    sql: str

@app.get("/estoque")
def get_estoque():
    """Rota GET para expor todo o banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM estoque").fetchall()
    conn.close()
    return {"data": [dict(row) for row in rows]}

@app.post("/query")
def execute_query(request: SQLRequest):
    """Rota POST que recebe SQL e devolve o resultado."""
    sql = request.sql.strip()
    
    # Checagem de segurança básica na API (redundância para o gerador)
    if not sql.upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Apenas consultas SELECT são permitidas.")
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        conn.close()
        return {"data": [dict(row) for row in rows]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"Erro ao executar SQL: {str(e)}")

if __name__ == "__main__":
    print("Iniciando Server na porta 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)