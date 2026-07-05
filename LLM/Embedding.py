# for Embeding model in endpoint

# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# load_dotenv()
# embedding = HuggingFaceEndpointEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2',
#                                           task='feature-extraction')
# text="Machine Learning is a field of AI "

# Text_embedding=embedding.embed_query(text)

# print(Text_embedding)

# Use Embed model locally

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Machine Learning is a field of AI "

Text_embedding = embedding.embed_query(text)

print(Text_embedding)

documents = ["Machine Learning is a field of AI ", "Deep Learning is a subset of Machine Learning", "Natural Language Processing is a field of AI"]
document_embeddings = embedding.embed_documents(documents)
print(document_embeddings)
