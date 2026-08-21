### nota: tive que instalar assim pip install dspy==3.3.0b1
### pois vi no github que soh essa beta release tem a classe SandBoxSerializable
### que faz o lance do dataframe 
### https://github.com/stanfordnlp/dspy/releases
import dspy
import pandas as pd
import fastparquet
from dspy.predict import RLM
from dataframe import DataFrame ### minha classe aqui desta pasta
from dspy_monty_interpreter import MontyInterpreter

interp = MontyInterpreter()



class DataAnalysisSignature(dspy.Signature):
    """Analyze complex multi-table datasets to answer business intelligence queries."""
    
    users: DataFrame = dspy.InputField(desc="User profiles with first_name, birthdate, salary, title")
    query: str = dspy.InputField(desc="The analytical question to resolve")
    
    insights: str = dspy.OutputField(desc="Detailed summary of findings and data trends")


users_df = pd.read_parquet("userdata1.parquet")

print(users_df.head())

### otima leitura https://dspy.ai/api/modules/RLM/
### https://kmad.ai/A-Data-Analysis-Agent-That-Lives-in-Your-Program
### https://github.com/kmad/dspy/blob/sandbox-serializable/docs/docs/tutorials/dataframe_rlm/cohort_analysis/cohort_analysis_demo.ipynb

wrapped_users = DataFrame(users_df)


lm = dspy.LM('openai/Qwen3_5-9B-Q4_1', api_base='http://localhost:1337/v1', api_key='not-needed')
dspy.configure(lm=lm)

analyst_agent = RLM(
    DataAnalysisSignature, 
    max_iterations=5, 
    max_llm_calls=12,
    interpreter=interp
)

# Run the query over the DataFrames
result = analyst_agent(
    users=wrapped_users,
    query="Find the customer cohort with the highest churn risk based on transaction volume dropping month-over-month."
)

print(result.insights)