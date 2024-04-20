from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key='')
pc = Pinecone(api_key="")
index = pc.Index("qa-embeddings")

system_role = """
당신은 네이버 스마트 스토어의 FAQ 챗봇입니다. 벡터 데이터베이스로부터 가져오는 질문,답변 쌍을 근거로 답변해야 하며, 유저의 이전 질문과 상황 등을 토대로 더 적절한 답변을 제공해야 합니다. 
[!IMPORTANT] 스마트스토어와 관련 없는 질문에는 답변하지 않아야 합니다. 부적절한 질문에는 "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."와 같이 안내 메시지를 출력합니다.
"""

message_history = [{'role':'system','content':system_role}]

print("\n무엇을 도와드릴까요? (그만하시려면 '끝'을 입력해주세요):\n")

try:
    first_time = True
    while True:
        if not first_time:
            print()
        first_time = False
        user_quest = input()

        # Exit loop if user types 'exit'
        if user_quest == '끝':
            break
            
        # Function to embed text
        def embed_text(text, model="text-embedding-3-small"):
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        
        # Function to generate answer
        def generate_answer(temp_message_history): 
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=temp_message_history,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.2,
            )
            return response.choices[0].message.content
        
        def transform_qa(input_dict):
            # Extract 'question' and 'answer' from the input dictionary
            question = input_dict.get('question', '')
            answer = input_dict.get('answer', '')
            
            # Create new dictionaries for user and assistant
            user_dict = {'role': 'user', 'content': question}
            assistant_dict = {'role': 'assistant', 'content': answer}
            
            # Return both dictionaries
            return user_dict, assistant_dict
        
        # Function to handle user's question
        def handle_question(user_quest):
            global message_history
            temp_message_history = message_history.copy()

            # Embed user's question
            user_question_embedding = embed_text(user_quest)
            results = index.query(
                top_k=5, 
                vector=user_question_embedding, 
                include_metadata=True,
            )

            for i in range(5):
                qa_dict = results['matches'][i]['metadata']
                user, assistant = transform_qa(qa_dict)
                temp_message_history.append(user)
                temp_message_history.append(assistant)
            
            temp_message_history.append({"role": "user", "content": user_quest})
           
            # Generate answer using OpenAI's LLM
            answer = generate_answer(temp_message_history)
        
            # Add the question and the generated answer to the message_history
            message_history.extend([{"role": "user", "content": user_quest}])
            message_history.extend([{"role": "assistant", "content": answer}])
        
            return answer
        
        answer = handle_question(user_quest)
        print(answer)

except Exception as e:
    print(e)
