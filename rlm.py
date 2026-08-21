import dspy
from dspy_monty_interpreter import MontyInterpreter

interp = MontyInterpreter()

lm = dspy.LM('openai/Qwen3_5-9B-Q4_1', api_base='http://localhost:1337/v1', api_key='not-needed')
dspy.configure(lm=lm)

### esse primeiro parametro abaixo eh um Signature simples sem class Signature
rlm = dspy.RLM("context -> answer", interpreter=interp)
result = rlm(context="what is 2 + 2?")
print(result)
print(result.trajectory[0]["code"]) ### ele chama o repl interpreter como tool