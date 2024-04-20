import pickle
from openai import OpenAI
from pinecone import Pinecone

with open('C:/Users/alsgo/OneDrive/바탕 화면/growth_hackers/coxwave_chatbot/final_result.pkl', 'rb') as f:
    faq_data = pickle.load(f)

client = OpenAI(api_key='')
pc = Pinecone(api_key="")
index = pc.Index("qa-embeddings")


# Function to embed text using OpenAI's text-embedding-3-small
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        )
    return response.data[0].embedding

# Processing Q&A data
for i, (question, answer) in enumerate(faq_data.items(), 1):
    combined_text = f"질문: {question} 답: {answer}"
    embedding = embed_text(combined_text)
    doc_id = f"id{i}"  
    
    # Upload to Pinecone with a unique ID
    index.upsert(vectors=[(doc_id, embedding, {"question": question, "answer": answer})])

print("Data has been embedded and uploaded to Pinecone successfully.")