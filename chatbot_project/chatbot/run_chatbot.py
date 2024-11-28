import pandas as pd
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


def embed(text):
  '''
  텍스트를 임베딩하는 함수
  Args:
    text (str) : 임베딩할 텍스트
  Returns:
    embedding (list) : 텍스트의 임베딩
  '''

  response = client.embeddings.create(input = text, model='text-embedding-3-small')
  embedding = response.data[0].embedding
  return embedding

def find_context(question,k,min_score):
  '''
  유저의 질문을 대답하기 위한 문맥을 찾는 함수
  Args:
    question (str) : 유저의 질문
    k (int) : 찾을 문맥의 수
    min_score (float) : 최소 유사도 점수
  Returns:
    context (str) : 상위 k 개의 문맥. 충분히 유사한 문맥이 없다면 '관련 없는 질문'임을 반환
  '''

  user_vector = embed(question)

  cosine_similarities = cosine_similarity(df['QA Vector'].tolist(), [user_vector])

  #가장 높은 유사도가 최소 요구 점수 미만이라면, '네이버 스마트 스토어와 관련 없는 질문'임을 반환
  if max(cosine_similarities) < min_score:
    context = '\n This question is unrelated to 네이버 스마트스토어'
    return context

  top_indices = cosine_similarities.flatten().argsort()[-k:][::-1]

  #상위 k개의 문맥을 하나의 str로 합치기
  context = ''
  for i in range(k):
      context += f'\n Context {i+1} \n'
      context += df['QA'].iloc[top_indices[i]]

  return context

def generate_question(message,history):
  '''
  지금까지의 대화와 새로운 유저 질문을 바탕으로 (문맥 검색에 사용할) 독립적인 질문을 생성하는 함수
  Args:
    message (str) : 유저의 새 질문
    history (str) : 지금까지의 대화 내용 (유저의 질문과 챗봇의 답변)
  Returns:
    question (str) : 생성된 독립적인 질문
  '''
  
  prompt = f'''Create a standalone question based on the chat history and the follow-up question. Return only the standalone question in Korean and no other explanation.
  Chat History: {history}
  Follow-up Question: {message}
  '''
  
  #GPT를 통해 새로운 질문 생성
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt}]
    )
  
  question = response.choices[0].message.content

  return question

def chatbot():
  
  #대화 시작
  intro = "저는 네이버 스마트스토어 챗봇입니다. 궁금한 점을 질문해주세요! (멈추려면 quit을 적어주세요)\n"
  print('챗봇:', intro)

  #대화 내역 기록 시작 (실제 유저 질문 및 챗봇 답변만 저장)
  history = 'assistant: ' + intro

  #GPT 답변 지시 prompt
  messages = [
    {"role": "system", "content": '''You are a helpful assistant that answers questions related to 네이버 스마트스토어. 
    For every Question, you will be given some Context that provides relevant information about 네이버 스마트스토어.
    You are only allowed to answer based on the given Context.
    If the Question is unrelated to 네이버 스마트스토어, answer with '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'
    Always answer in Korean.'''},
  ]

  while True:
    # 질문 입력
    message = input("유저: ")

    # 유저가 quit을 입력하면 중단
    if message.lower() == "quit":
      break

    # 지금까지 질문이 두 개 이상 있었다면, 대화 내역을 바탕으로 독립적인 질문을 먼저 생성한 뒤 진행
    if len(messages)>2:
      message = generate_question(message, history)

    #답변에 활용할 문맥 검색
    context = find_context(message,3,0.4)

    #GPT에게 질문 및 문맥 제공
    messages.append({"role": "user", "content": f"Question: {message} {context}"})
    history += f'user: {message} /n '

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    # 답변 출력
    chat_message = response.choices[0].message.content
    print(f"\n챗봇: {chat_message}\n")
    messages.append({"role": "assistant", "content": chat_message})
    history += f'assistant: {chat_message} /n '


if __name__ == "__main__":

  #OpenAI API 설정
  api_key = input('Input API KEY: ')
  os.environ["OPENAI_API_KEY"] = api_key
  client = OpenAI()

  #Vector DB 불러오기
  df = pd.read_pickle('chatbot/vectordb.df')

  #챗봇 실행
  chatbot()
