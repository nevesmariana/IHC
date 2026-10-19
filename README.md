# IHC - Consulta de estoque por linguagem natural

Bot do Telegram que recebe perguntas sobre um estoque,
gera um SQL com DSPy, consulta uma API FastAPI e responde em formato json.

## Requisitos
- Python 3.10+
- Servidor LLM local compatível com OpenAI em `http://localhost:1337/v1`
  (modelo `gemma-4-E2B-it-IQ4_XS`)
- Token de bot do Telegram (via @BotFather)

## Instalação
```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pydantic pyTelegramBotAPI requests openai-whisper dspy
```

## Como rodar
1. Inicie o servidor de LLM local na porta 1337.
2. Em um terminal: `python server.py`
3. Em outro terminal: `export TELEGRAM_TOKEN="..."` e depois `python bot.py`
4. Converse com o bot no Telegram.

## Exemplo de uso
- "Quantos sabonetes temos?"
- "Qual a validade da coca-cola?"