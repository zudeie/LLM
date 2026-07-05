from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Machine Learning is a field of AI "

Text_embedding = embedding.embed_query(text)

# print(Text_embedding)

documents = ["Machine Learning is a field of AI ", "Deep Learning is a subset of Machine Learning", "Natural Language Processing is a field of AI"]
document_embeddings = embedding.embed_documents(documents)
# print(document_embeddings)

query=" what is Natural Language Processing?"
query_embedding=embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)
print(scores)
index, score = sorted(
    enumerate(scores[0]),
    key=lambda x: x[1]
)[-1]
print(f"Most relevant document: {documents[index]} with score: {score}")