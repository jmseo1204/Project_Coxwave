import pickle

with open("../final_result.pkl", "rb") as f:
    docs = pickle.load(f)
len(docs)

import os
from openai import OpenAI
import openai
import json

# 보안 상 git push 안돼서 주석처리
API_KEY = os.environ["OPENAI_API_KEY"]


def gpt(prompt: list, history=[]):
    llm = OpenAI(api_key=API_KEY)
    ans_JSON = []

    history += prompt

    respond = llm.chat.completions.create(
        messages=history,
        model="gpt-3.5-turbo",
        # response_format= {"type":"json_object"},
        temperature=0.5,
    )
    history.append({"role": "assistant", "content": respond.choices[0].message.content})
    # ans_JSON.append(json.loads(respond.choices[0].message.content or {}))
    print(history[-1]["content"])
    return history


history = []
gpt([{"role": "user", "content": "삼행시 지어줘"}])

import chromadb
import time
import pandas as pd
import numpy as np

# from langchain.embeddings import CacheBackedEmbeddings
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings

embedded_data = pd.read_csv("../embeddings.csv")

checkpoint = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

client = chromadb.PersistentClient("../")


collection = client.get_or_create_collection(
    name="QnA_data", metadata={"hnsw:space": "cosine"}
)

embedded = {"embeddings": [], "ids": []}
for iter in embedded_data[["question_vector", "answer_vector"]].iterrows():
    doc = iter[1]["question_vector"] + iter[1]["answer_vector"]
    # 개선사항 -> Q, A 각각 임베딩을 각각하는 것보다 두 개를 [SEP]으로 이어붙여서 한 문장으로 만들면 더이상 질문과 응답을 나눠 생각 안해도 된다.
    # print(type(iter[1]['question_vector']))
    embedded["embeddings"].append(doc)
    embedded["ids"].append(str(iter[0]))

# make vectorDB with embedding

collection.add(embeddings=embedded["embeddings"], ids=embedded["ids"])

import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


history = []
while True:
    question = input("Ask to Naver Smartstore Chatbot: ")
    if question == "":
        break

    encoded_input = tokenizer(
        question,
        padding="max_length",
        max_length=1536,
        truncation=False,
        return_tensors="pt",
    )
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(
        model_output, encoded_input["attention_mask"]
    ).tolist()
    print(len(sentence_embeddings), sentence_embeddings)
    related = collection.query(query_embeddings=sentence_embeddings, n_results=5)

    map_prompt_system = """
    You are information giver and user want to get the related useful answer. 
    Answer the question regarding below QnA history. If there is not related information, just return the below the 'QnA history' sentences straightforward. 

    [QnA history] : {context}
    """
    reduced = []
    for doc in related:
        map_prompt = [
            {"role": "system", "content": map_prompt_system.format(context=doc)},
            {"role": "user", "content": question},
        ]
        reduced.append(gpt(map_prompt)[-1]["content"])
    print(reduced)
    final_prompt_system = """
    You are 네이버 스마트스토어 Chatbot to give a right answer according to the background information. 
    User want to get the right and objective answers in Korean, so you must reply with Korean sentence.
    Here are some appropriate examples. You should answer like these.

    [good conversation examples] :
    user : 미성년자도 판매 회원 등록이 가능한가요?
    chatbot(you) : 
        네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
    user : 저는 만 18세입니다.
    chatbot(you) :
        만 14세 이상 ~ 만 19세 미만인 판매회원은 아래의 서류를 가입 신청단계에서 제출해주셔야 심사가 가능합니다.
        (관련된 추가 설명)
    user : 민증이 없어요.
    chatbot(you) : 만 18세 회원님의 경우 가입심사 서류만 제출해주시면 됩니다.


    And keep in mind that if there is no useful information or user gave you an improper question \
        which is unrelated with '네이버 스마트스토어' , just reply '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.' mechanically.
    Here is related example.

    [example with improper question from user]
    user : 오늘 저녁에 여의도 가려는데 맛집 추천좀 해줄래?
    chatbot(you) : 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.


    Lastly, Here is background information you must utilize. 
    Consider below ones comprehensively and answer the question. 
`
    [background information] : {context}

    """

    final_prompt = [
        {
            "role": "system",
            "content": final_prompt_system.format(context="\n\n".join(reduced)),
        },
        {"role": "user", "content": question},
    ]
    history = gpt(final_prompt, history)
