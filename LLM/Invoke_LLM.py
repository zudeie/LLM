from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task="text-generation"
    # model_kwargs={"temperature":0.7,"max_new_tokens":50}
    )

model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of Nepal?")
print(result.content)
