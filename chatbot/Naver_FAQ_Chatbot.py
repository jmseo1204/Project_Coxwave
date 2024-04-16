from openai import OpenAI
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import os
from ast import literal_eval
import warnings



# naver FAQ 데이터 처리 함수
def create_dataframe(dictionary):
    data = []
    for id, (question, answer) in enumerate(dictionary.items()):
        question = question.replace("\n", " ")
        answer = answer.replace("\xa0", "")
        answer = answer.replace("\n"," ")
        answer = answer.replace(" 위 도움말이 도움이 되었나요?   별점1점  별점2점  별점3점  별점4점  별점5점    소중한 의견을 남겨주시면 보완하도록 노력하겠습니다.  보내기 ","")
        answer = answer.replace(" 도움말 닫기","")
        data.append([id+1, question, answer])
    df = pd.DataFrame(data, columns=['id', 'question', 'answer'])
    return df


# answer 길이가 긴 데이터프레임 처리 함수
def split_long_answers(df, max_length=512):
    new_rows = []
    total_rows = len(df)
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for index, row in df.iterrows():
            answer = row['answer']
            if len(answer) > max_length:
                num_splits = len(answer) // max_length + 1
                for i in range(num_splits):
                    start_idx = i * max_length
                    end_idx = (i + 1) * max_length
                    new_answer = answer[start_idx:end_idx]
                    new_row = row.copy()  # 기존 행을 복사하여 수정
                    new_row['answer'] = new_answer
                    new_rows.append(new_row)
                pbar.update(1) 
            else:
                new_rows.append(row)  # 'answer' 열의 길이가 max_length 이하인 경우에는 기존 행 그대로 추가
                pbar.update(1)  
    df = pd.DataFrame(new_rows)
    return df

# 임베딩 생성 함수
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    outputs = model(**inputs)
    return outputs.last_hidden_state[0].detach().numpy()[0].tolist()

# 데이터프레임에 임베딩 column 생성 함수
def add_embeddings_to_dataframe(df):
    tqdm.pandas(desc="Calculating question embeddings")
    df['question_vector'] = df['question'].progress_apply(lambda x: get_embedding(x))
    
    tqdm.pandas(desc="Calculating answer embeddings")
    df['answer_vector'] = df['answer'].progress_apply(lambda x: get_embedding(x))
    
    return df

# 임베딩을 리스트 형태로 반환해주는 함수
def embedding_function(input):
    texts = input['texts']
    return [get_embedding(text) for text in texts]

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        texts = input  # 텍스트 리스트
        embeddings = [get_embedding(text) for text in texts]
        return embeddings

# chromadb에 데이터 삽입 함수
def insert_data(df):
    chroma_client = chromadb.PersistentClient(path="./naver_data.db")

    chroma_client.delete_collection(name='answer')
    chroma_client.delete_collection(name='question')
    
    answer_collection = chroma_client.create_collection(name='answer', metadata={"hnsw:space": "cosine"}, embedding_function=MyEmbeddingFunction())
    question_collection = chroma_client.create_collection(name='question', metadata={"hnsw:space": "cosine"}, embedding_function=MyEmbeddingFunction())
    
    answer_collection.add(
        ids=df.id.tolist(),
        embeddings=df.answer_vector.tolist(),
    )
    
    # Add the title vectors
    question_collection.add(
        ids=df.id.tolist(),
        embeddings=df.question_vector.tolist(),
    )
    
    return question_collection, answer_collection

# 쿼리 생성함수
def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    return results

# context 생성 함수
def build_context(df, question, question_collection, answer_collection):
    context = ""
    
    result = query_collection(question_collection, [question], 2, df)
    searched_ids = result['ids'][0]
    searched_answers = (df[df['id'].isin(searched_ids)]['question'] + df[df['id'].isin(searched_ids)]['answer']).tolist()
    
    result2 = query_collection(answer_collection, [question], 1, df)
    searched_ids2 = result2['ids'][0]
    searched_answers2 = (df[df['id'].isin(searched_ids2)]['question'] + df[df['id'].isin(searched_ids2)]['answer']).tolist()
    
    searched_answers += searched_answers2

    for doc in searched_answers:
        context += str(doc) + '\n\n'
    return context
    

# 답변해주는 함수
def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    print(result.choices[0].message.content + '\n')
    return result.choices[0].message.content



def chatbot(df, question, question_collection, answer_collection):
    global history
    
    context = build_context(new_df, question, question_collection, answer_collection)

    # 시스템 메시지에 이전 대화 내용을 포함하여 생성
    system = f"""너는 한국 사람들이 가장 많이 이용하는 포털사이트 네이버의 서비스 스마트스토어에서 근무하는 도우미야. 
    스마트스토어에 입점하고자 하는 사람들의 질문을 받거나, 스마트스토어 규정 관련해서 아주 잘 알고있는 도우미지. 
    너는 사용자의 질문에 정확하고 친절하게 답해야해.

    [good example]
    유저 : 미성년자도 판매 회원 등록이 가능한가요?
    챗봇 : 
        네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
    유저 : 저는 만 18세입니다.
    챗봇 :
        만 14세 이상 ~ 만 19세 미만인 판매회원은 아래의 서류를 가입 신청단계에서 제출해주셔야 심사가 가능합니다.
        추가 설명 ~~

    [imappropriate question example]
    유저 : 오늘 저녁에 여의도 가려는데 맛집 추천좀 해줄래?
    챗봇 : 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.

    만약 질문이 모호하다면 한 번 더 질문할 수 있어
    [good example]
    유저 : 절차를 알려줘
    챗봇 : 무슨 절차를 안내해드릴까요?
    유저 : 회원가입 절차를 알려줘
    챗봇 : 
    네이버 스마트스토어 회원가입 절차는 다음과 같아요:
    1. 네이버 아이디로 로그인
    2. 스마트스토어 이용약관 동의
    3. 사업자 정보 입력
    4. 상품 및 쇼핑몰 정보 입력
    5. 결제정보 입력
    6. 가입 완료
    
    자세한 내용은 네이버 스마트스토어 홈페이지에서 확인하실 수 있어요. 추가로 궁금한 점이 있으면 언제든지 물어봐주세요.

    너는 반드시 관련된 문맥(context)를 기반으로 대답해야해. 질문의 문맥은 아래와 같아
    그리고 반드시 이전에 했던 대화를 기반으로 대답해야해
    [context] = {context}
    """
    
    message = [{'role':'system','content':system}]
    
    if len(history) != 0:
        for m in history[0]:
            if m['role'] == 'user':
                message.append(m)
            if m['role'] == 'assistant':
                message.append(m)

    # 사용자 질문 메시지 추가
    user_message = {'role': 'user', 'content': question}
    message.append(user_message)

    answer = generate_response(message)

    history.append(message)
    history.append({'role': 'assistant', 'content': answer})
    
    return answer



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    OPEN_API_KEY = input("Open AI API key를 입력해 주세요: ")
    client = OpenAI(api_key=OPEN_API_KEY)
    naver_data = pd.read_pickle("final_result.pkl")
    
    # naver FAQ 데이터 처리
    df = create_dataframe(naver_data)
    new_df = split_long_answers(df, max_length=512)

    model_checkpoint = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)

    if not os.path.exists('embeddings.csv'):
        print('embeddings.csv 파일이 없으므로 임베딩을 진행합니다')
        df_with_embeddings = add_embeddings_to_dataframe(new_df)
        df_with_embeddings.to_csv('new_embeddings.csv')
        new_df = df_with_embeddings.copy()
        new_df.reset_index(drop=True, inplace=True)
        new_df['id'] = new_df.index
        new_df['id'] = new_df['id'].astype(str)
        
    else:
        print('embeddings.csv 파일이 있으므로 넘어갑니다')
        new_df = pd.read_csv('embeddings.csv')
        new_df.reset_index(drop=True, inplace=True)
        new_df['id'] = new_df.index
        new_df['id'] = new_df['id'].astype(str)
        new_df['question_vector'] = new_df['question_vector'].apply(literal_eval)
        new_df['answer_vector'] = new_df['answer_vector'].apply(literal_eval)

    print('chroma db에 데이터 삽입')
    question_collection, answer_collection = insert_data(new_df)

    history = []
    print('====================================================')
    print("챗봇 시작. 무엇을 도와드릴까요? 종료를 원하시면 '종료'를 입력하세요.")

    while True:
        question = input('질문 입력: ')
        if question == '종료':
            print('====================================================')
            print("챗봇 종료.")
            break
        chatbot(new_df, question, question_collection, answer_collection)