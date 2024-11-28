import os

from dotenv import load_dotenv
import openai
from openai import OpenAI
from pinecone import Pinecone
import yaml

from Project_Coxwave.Chatbot.characters import main_system_role
from Project_Coxwave.Chatbot.input_command_checker import Command, get_command

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))


chatbot_params_path = 'C:/Users/alsgo/OneDrive/바탕 화면/growth_hackers/Project_Coxwave/chatbot_params.yaml'
with open(chatbot_params_path, "r") as file:
    params = yaml.safe_load(file)


Message_history = [{'role':'system','content':main_system_role}]


def split_question_answer(input_dict):
    # Extract 'question' and 'answer' from the input dictionary
    question = input_dict.get('question', '')
    answer = input_dict.get('answer', '')
    
    # Create new dictionaries for user and assistant
    user_dict = {'role': 'user', 'content': question}
    assistant_dict = {'role': 'assistant', 'content': answer}
    
    # Return both dictionaries
    return user_dict, assistant_dict


def embed_text(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def generate_answer(message_history): 
    params["messages"] = message_history
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def handle_question(user_quest):
    temp_message_history = Message_history.copy()

    # Embed user's question
    user_question_embedding = embed_text(user_quest)
    
    # Extract the 3 q&a vectors that are most similar to the user's question embedding
    index = pc.Index("qa-embeddings")
    results = index.query(
        top_k=3, 
        vector=user_question_embedding, 
        include_metadata=True,
    )
    # Make 3 q&a as if the user has asked and been answered before.
    for i in range(3):
        qa_dict = results['matches'][i]['metadata']
        user, assistant = split_question_answer(qa_dict)
        temp_message_history.append(user)
        temp_message_history.append(assistant)
    
    temp_message_history.append({"role": "user", "content": user_quest})
    
    # Generate answer using OpenAI's LLM
    answer = generate_answer(temp_message_history)

    # Add the question and the generated answer to the message_history
    Message_history.extend([{"role": "user", "content": user_quest}])
    Message_history.extend([{"role": "assistant", "content": answer}])

    return answer


if __name__ == "__main__":
    try:
        print("\n무엇을 도와드릴까요? (그만하시려면 '끝'을 입력해주세요):\n")
        while True:
            user_quest = input()
            command = get_command(user_quest)
        
            if command == Command.END:
                print("챗봇을 종료하겠습니다.")
                break  
                
            answer = handle_question(user_quest)
            print(answer)
            
    except openai.error.InvalidRequestError as e:
        # Handle Tokens Exceeded Error
        if "That model's maximum context length is" in str(e):
            print("오류: 입력 프롬프트 또는 생성된 출력이 최대 허용 토큰 수를 초과했습니다.\n질문을 다시 표현하거나 작은 부분으로 나누어 시도해 주세요.")
        else:
            print(f"오류: {e}\n죄송합니다. 요청을 처리하는 동안 오류가 발생했습니다.")
            
    except openai.error.AuthenticationError:
        # Handle Authentication Error
        print("오류: 잘못된 또는 누락된 API 키입니다.")

    except:
        print("예상치 못한 오류가 발생했습니다.")
