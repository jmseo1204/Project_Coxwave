import pickle
import pandas as pd
import numpy as np
import re
from openai import OpenAI
import os
from tqdm import tqdm
import time

def clean(answer):
    '''
    주어진 답변을 전처리하는 함수
    Args:
        answer (str) : FAQ의 답변 데이터
    Returns:
        answer (str) : 전처리된 답변 데이터
        related (list) : 관련 도움말 목록
    '''

    #불필요한 별점 제거
    other = '\n\n\n위 도움말이 도움이 되었나요?\n\n\n별점1점\n\n별점2점\n\n별점3점\n\n별점4점\n\n별점5점\n\n\n\n소중한 의견을 남겨주시면 보완하도록 노력하겠습니다.\n\n보내기\n\n\n\n'
    answer = answer.replace(other, ' ')
    answer = answer.replace('도움말 닫기', '')

    #관련 도움말은 따로 저장
    if '관련 도움말/키워드' in answer:
        a = answer.split('관련 도움말/키워드')
        answer = a[0]
        related = a[1]
        related=related.strip().split('\n')

    else:
        related = np.nan

    #전처리
    answer = re.sub('\s{2,}', '\n', answer) #빈칸이 긴 경우 줄이기
    answer = answer.replace('\xa0','\n') #\xa0를 \n으로 통일
    answer = answer.replace("\'",'') #볼드 표시 제거

    return answer, related

def get_category(text):
    '''
    질문의 카테고리를 추출하는 함수
    Args:
        text (str) : FAQ의 질문 데이터
    Returns:
        extracted_text (str) : 질문이 속하는 카테고리 (없다면 np.nan 반환)
    '''
    pattern = r'^\[(.*?)\]'  #[]표시로 시작하는 경우 카테고리로 인식
    matches = re.findall(pattern, text)
    if matches:
        extracted_text = matches[0]  #첫 카테고리만 반환
        return extracted_text
    else:
        return np.nan


def chunk_string(long_string, chunk_size=500):
    '''
    긴 문자열을 짧게 나누는 함수 
    Args:
        long_string (str) : 긴 문자열
        chunk_size (int) : 원하는 문자열 길이
    Returns:
        chunks (list) : 나눠진 짧은 문자열의 리스트
    '''
    chunks = []
    current_chunk = ''
    last_line = ''
    previous_line = ''

    for line in long_string.split('\n'):
        # 주어진 chunk_size를 초과하지 않는 한 계속해서 현재 chunk에 새로운 줄 추가
        if len(current_chunk) + len(line) <= chunk_size:
            current_chunk += line + '\n'
            last_line = line + '\n'
        # 현재 청크에 줄을 추가하면 chunk_size를 초과하게 되면 현재 청크 완성
        else:
            #이전 chunk의 마지막 줄도 overlap되게 앞에 포함
            chunks.append((previous_line+current_chunk).rstrip('\n'))
            previous_line = last_line
            current_chunk = line + '\n'

    # 마지막으로 남은 chunk 처리
    if current_chunk:
        chunks.append((previous_line+current_chunk).rstrip('\n'))

    return chunks

def embed(text, wait_time=0.1):
   '''
    텍스트를 임베딩하는 함수
    Args:
        text (str): 임베딩할 텍스트
        wait_time (float): 요청 사이의 간격
    Returns:
        embedding (list): 텍스트의 임베딩
    '''
   response = client.embeddings.create(input = text, model='text-embedding-3-small')
   time.sleep(wait_time)
   embedding = response.data[0].embedding
   return embedding

if __name__=="__main__":
  
  #FAQ 데이터 불러오기
  with open('final_result.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

  df = pd.DataFrame(loaded_data.items())
  df.columns = ['Question','Answer']

  #질문의 카테고리 정보 추출
  df['Category'] = df['Question'].apply(get_category)

  #답변 데이터 전처리 및 관련 질문 정보 추출
  df['Related'] = np.nan
  df[['Answer', 'Related']] = df['Answer'].apply(clean).apply(pd.Series)

  #임베딩 설정
  api_key = input('Input API KEY: ')
  os.environ["OPENAI_API_KEY"] = api_key
  client = OpenAI()

  #질문 임베딩
  tqdm.pandas(desc='Embedding Questions')
  df['Question Vector'] = df['Question'].progress_apply(embed)

  #답변 chunking
  df['Answer']=df['Answer'].apply(chunk_string)
  df = df.explode('Answer')
  df = df.reset_index()
  df = df.rename(columns={'index': 'Question Index'})

  #답변 임베딩
  tqdm.pandas(desc='Embedding Answers')
  df['Answer Vector'] = df['Answer'].progress_apply(embed)

  #질문-답변 쌍 임베딩
  df['QA'] = df['Question'] + ' ' + df['Answer']
  tqdm.pandas(desc='Embedding Question-Answer Pairs')
  df['QA Vector'] = df['QA'].progress_apply(embed)

  #최종 벡터 DB 저장
  df.to_pickle('vectordb.df')