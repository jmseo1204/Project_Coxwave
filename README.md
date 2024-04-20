[Reference]

1) tokenizing & embedding

hugging face Question-answering method
https://huggingface.co/learn/nlp-course/chapter7/7
https://huggingface.co/docs/transformers/tasks/question_answering

fine-tunning 
(without any code!) https://huggingface.co/blog/Llama2-for-non-engineers


summerization methods(stuff, map-reduce, refine)
https://python.langchain.com/docs/use_cases/summarization/

principle of attention technique and transformer(blog series)
https://tigris-data-science.tistory.com/entry/DL-%EC%89%BD%EA%B2%8C-%ED%92%80%EC%96%B4%EC%93%B4-Attention-Mechanism-1-Bahdanau-Attention


[Workflow]
1) 처음엔 gpt, vectorDB 만으로 문장유사도를 통해 관련 질의응답 데이터를 뽑아서 context로 LLM에게 넘겨주는 방식
-> 이는 엄청나게 긴 QnA문장을 하나의 임베딩으로(그것도 범용적인 pretrained사용) 변환하기에 부정확할 수 밖에 없음. 최소한 문장 tokenizer 필요
2) fine-tunning에 대한 로망이 있었기에 huggingface의 question answering task 구현 과정을 공부하고 따라함.
-> 여기서 모델 train시 질문에 대한 정답 데이터가 context데이터의 몇 번 인덱스에 있는지가 label로 제공된다고 가정하는데, 우리 데이터에는 그런 정답 label이 없음.(아..) 
그런데 적어도 이렇게 QnA만을 위해 만들어진 모델은 단순히 gpt에 때려넣는 모델보다 더 정확한 답변을 제공할 것으로 추정
-> huggingface 문서를 찾아보다가 다음을 발견했다. (https://huggingface.co/tasks/question-answering)

There are different QA variants based on the inputs and outputs:

Extractive QA: The model extracts the answer from a context. The context here could be a provided text, a table or even HTML! This is usually solved with BERT-like models.
Open Generative QA: The model generates free text directly based on the context. You can learn more about the Text Generation task in its page.
Closed Generative QA: In this case, no context is provided. The answer is completely generated by a model.

쉬운 방식인 gpt에 프롬프트 형식으로 context 때려넣는 건 generarive QA 방식인듯. 사실 베스트는 데이터에 answer label이 있어서 fine-tunning하는 거지만, 일단 안되니까 extractive QA로 task 해결하겠다. 어쩌면 기본 bert 말고 다른 모델은 fine tuning을 정답 label 없이도 할 수 있을지도.. 근데 마감까지 시간이 부족하다. 일단 은희가 embedding한 csv 데이터를 사용해야만 한다. 어차피 cos sim 이용하니까 tokenizer만 같은 checkpoint 사용하고, 관련 context에서 answer 추출할 때는 SOTA에 가까운 모델을 사용하는 게 좋을 것 같다.

checkpoint - jhgan/ko-sroberta-multitask 문서를 확인해보면 kort를 sequencetransformer로 training 시킨 모델이다. 이런.. 애초에 loss를 cos simul로 잡은 애라 진짜 문장유사도만 출력해내는 모델이다. 이 모델로 question answering task를 해낼 수 있을지 미지수. 일단 pipeline을 써보자. 



3) 따라서 프롬프팅과 QnA모델을 적절히 병합해야하는데, 구상한 프로세스는 다음과 같다.
user query -> [질문 적합성 판단 - gpt를 통해 네이버 스마트스토어와 부적합한 질문은 거절하기] -> [쿼리와 context 유사도 판단 - 이는 코사인유사도로 뽑아낼 수 밖에 없다. ] 