# IHC - Consulta de estoque por linguagem natural

Bot do Telegram que recebe perguntas em português sobre um estoque, transforma a
pergunta em um comando SQL usando um modelo de linguagem local (Gemma) com o
framework **DSPy**, consulta uma API **FastAPI** e devolve o resultado em JSON.

## Requisitos

- Python 3.10 ou superior
- Servidor de LLM local compatível com a API da OpenAI (usamos o **Jan**) em
  `http://localhost:1337/v1`, com o modelo `gemma-4-E2B-it-IQ4_XS` carregado
- Token de um bot do Telegram (crie pelo **@BotFather**)

## Instalação

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\activate

pip install fastapi uvicorn pyTelegramBotAPI requests dspy litellm
```

> O `litellm` é necessário porque o `bot.py` envia uma opção extra ao modelo
> (desligar o modo de raciocínio do Gemma). Sem ele, toda chamada ao modelo falha.

## Como rodar

1. Abra o Jan, carregue o modelo `gemma-4-E2B-it-IQ4_XS` e ligue o servidor local
   na porta 1337.
2. Em um terminal, suba a API:

   ```bash
   python server.py
   ```

3. Em **outro** terminal (com o venv ativado), defina o token e rode o bot:

   ```bash
   # Linux / Mac
   export TELEGRAM_TOKEN="seu_token_aqui"

   # Windows (PowerShell)
   $env:TELEGRAM_TOKEN="seu_token_aqui"

   python bot.py
   ```

4. O bot pergunta se você quer otimizar o prompt antes de iniciar (veja abaixo).
   Para só usar o bot, responda `n`.
5. Converse com o bot no Telegram.

> **Não coloque o token no código nem no GitHub.** Use sempre a variável de ambiente.

## Exemplos de uso

Pergunta: `Quantos sabonetes temos?`

```json
{
  "produto": "sabonete",
  "quantidade": 120
}
```

Pergunta: `Qual a validade da coca cola?`

```json
{
  "produto": "coca-cola lata",
  "data_validade": "2026-05-15"
}
```

Outras perguntas que o bot entende:
- `Qual o departamento do detergente?`
- `Quantas águas temos?`
- `Quais produtos ja venceram?`

Quando a resposta tem uma linha, o bot devolve um objeto; quando tem várias, devolve uma lista.
Se algo der errado (SQL inválido, API fora do ar), a resposta é `{"erro": "..."}`.