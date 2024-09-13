import openai
import asyncio
import re

openai.api_key = 'xxxxxx'
openai.base_url = "http://localhost:1234/v1/"

class OpenAIHandler:
    def __init__(self):
        self.chat_history = [{"role": "system", "content": "你是聪明的人工智能，你的话语简洁，字符使用简体中文，不加任何特殊字符。"}]

    def add_to_history(self, ai_response):
        self.chat_history.append({"role": "assistant", "content": ai_response})

        if len(self.chat_history) > 5:
            self.chat_history.pop(1)  


    def get_openai_response(self, prompt, output_queue):
        """
        与OpenAI API交互，并将结果放入队列中。
        
        :param self: 
        :param prompt: 
        """
        self.chat_history.append({"role": "user", "content": prompt})
            
        ai_response = ""
        current_sentence = ""

        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=self.chat_history,
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content or ""
                ai_response += delta
                current_sentence += delta
                print(delta,end="",flush=True)
                # 句子断句逻辑
                if len(current_sentence) > 4 and re.search(r'[.!?；：！？。]\s*|\n', current_sentence[5:]):
                    match = re.search(r'(.*?)([.!?；：！？。])\s*', current_sentence)
                    if match:
                        sentence = match.group(0).strip()
                        output_queue.put(sentence.replace("*",""))
                        current_sentence = current_sentence[len(sentence):].strip()

        # 如果最后一句话没有结束符，可以在这里处理
        if current_sentence:
            sentence = current_sentence.strip()
            output_queue.put(sentence.replace("*",""))

        # 添加完整的历史记录
        self.add_to_history(ai_response)


