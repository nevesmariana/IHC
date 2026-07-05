import dspy

lm = dspy.LM('openai/Qwen3_5-9B-Q4_1', api_base='http://localhost:1337/v1', api_key='not-needed')
dspy.configure(lm=lm)

class SimpleQA(dspy.Signature):
    ''''Answer questions briefly'''
    question = dspy.InputField()
    answer = dspy.OutputField()

predict = dspy.Predict(SimpleQA)
response = predict(question="Who are you?")
print(response.answer)