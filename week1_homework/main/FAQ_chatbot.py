import pickle
import pandas as pd
import chromadb
import os
from openai import OpenAI
from tqdm import tqdm

# Knowledge base 전처리
with open('main/final_result.pkl', 'rb') as file:
    data = pickle.load(file)

for key in data:
    data[key] = data[key].split('\n', 1)[0]

df = pd.DataFrame(columns=['question', 'answer'])
for idx, key in enumerate(data):
    df.loc[idx] = [key, data[key]]

# Knowledge base embedding
chroma_client = chromadb.Client()

question_collection = chroma_client.create_collection(name="question")
answer_collection = chroma_client.create_collection(name="answer")

os.environ["OPENAI_API_KEY"] = ""
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(
        input=[text],
        model=model
    )

    return response.data[0].embedding

def split_text(text, max_length):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    question_embedding = get_embedding(row["question"])
    question_collection.add(
        ids=[str(idx)], 
        embeddings=[question_embedding]
    )

    max_length = 1536
    if len(row["answer"]) > max_length:
        splitted_answers = split_text(row["answer"], max_length)
        for part_idx, part in enumerate(splitted_answers):
            part_embedding = get_embedding(part)
            answer_collection.add(
                ids=[f"{idx}_{part_idx}"],
                embeddings=[part_embedding]
            )
    else:
        answer_embedding = get_embedding(row["answer"])
        answer_collection.add(
            ids=[str(idx)],
            embeddings=[answer_embedding]
    )

# Query embedding + Similarity Search
def get_advanced_query(messages, query):
    prompt_parts = [
        'The following is a conversation record of a chatbot designed to resolve inquiries from merchants using an e-commerce platform called "스마트스토어":\n\n'
    ]
    
    for message in messages:
        if message["role"] == "user":
            prompt_parts.append(f'user: {message["content"]}\n')
        elif message["role"] == "assistant":
            prompt_parts.append(f'assistant: {message["content"]}\n\n')
    
    prompt_parts.append(f'After this, the user asked a new question:\n{query}\n\n')
    prompt_parts.append("Based on the chatbot's conversation history and the user's new question, identify what the user is curious about and refine the question into a single, detailed, interrogative sentence. You must answer in Korean.")

    return ''.join(prompt_parts)

def get_evidence(messages, query):
    advanced_query = ""
    if len(messages) == 1:
        advanced_query = query
    else:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": get_advanced_query(messages, query),
                },
            ],
            stream=False,
        )
        advanced_query = response.choices[0].message.content
    
    query_embedding = get_embedding(advanced_query)

    for collection in [question_collection, answer_collection]:
        vector_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        ids = vector_result['ids'][0]
        evidence = ""
        for id in ids:
            if '_' in id:
                id = id.split('_')[0]
            row = df[df.index == int(id)]
            question = row['question'].values[0]
            answer = row['answer'].values[0]
            evidence += f'question: {question}\nanswer: {answer}\n\n'

    return evidence

# Prompt 작성
system_prompt = """
    You are a chatbot dedicated to assisting merchants using an e-commerce platform called "스마트스토어". When generating responses, you must follow these rules:

    1. Answer in Korean.
    2. Your knowledge base consists of FAQ data from "스마트스토어". Each time, you base your response on provided FAQ question-answer pairs. If there are duplicate pairs, it is more likely that the pairs will contain an appropriate answer for the question. 
    3. Rather than merely quoting FAQ data, sift through and blend the information to create a natural and concise response that aligns with the intent of the question.
    4. If the user asks follow-up questions, consider both the FAQ question-answer pairs and the context of the previous conversation when generating your response.
    5. If the FAQ data is completely unrelated to the question, produce the following response exactly as it is written:
    저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.

    Here are some good examples:

    [example 1]
    user: 미성년자도 판매 회원 등록이 가능한가요?
    assistant: 네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.

    [example 2]
    user: 오늘 저녁에 여의도 가려는데 맛집 추천좀 해줄래?
    assistant: 저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.
"""

# LLM Answer 받기
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
]

def get_response(query):
    global messages

    messages_temp = messages[:]
    messages_temp.extend([
        {
            "role": "system",
            "content": get_evidence(messages, query),
        },
        {
            "role": "user",
            "content": query,
        }
    ])

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_temp,
        stream=True,
    )

    first_chunk = True
    response_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            if first_chunk:
                print(f"챗봇: {chunk.choices[0].delta.content}", end="")
                first_chunk = False
            else:
                print(chunk.choices[0].delta.content, end="")
            response_message += chunk.choices[0].delta.content

    if response_message != "저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.":
        messages.extend([
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": response_message,
            }
        ])

    return response_message
    
# Chatbot 실행
while True: # "종료"를 입력하면 대화 종료
    user_query = input("유저:")
    if user_query == "종료":
        break
    get_response(user_query)