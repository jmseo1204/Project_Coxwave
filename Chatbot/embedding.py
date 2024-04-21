import os

from dotenv import load_dotenv
from openai import OpenAI
import pickle
from pinecone import Pinecone

final_result_path = 'C:/Users/alsgo/OneDrive/바탕 화면/growth_hackers/coxwave_chatbot/final_result.pkl'
with open(final_result_path, 'rb') as f:
    faq_data = pickle.load(f)

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("qa-embeddings")

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
    
    # Upload to Pinecone with a unique ID and metadata consisting of a question and the answer
    index.upsert(vectors=[(doc_id, embedding, {"question": question, "answer": answer})])

print("Data has been embedded and uploaded to Pinecone successfully.")