import openai
import asyncio
import re

openai.api_key = 'xxxxxx'
openai.base_url = "http://localhost:1234/v1/"

class OpenAIHandler:
    def __init__(self):
        self.chat_history = []

    def add_to_history(self, user_input, ai_response):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": ai_response})

    # def get_openai_response(self, prompt):
    #     self.chat_history.append({"role": "user", "content": prompt})
        
    #     response = openai.chat.completions.create(
    #         model="gpt-3.5-turbo",  # 修改为合适的模型名
    #         messages=self.chat_history,
    #         stream=True,  # 使用流模式
    #         temperature=0.7,
    #     )

    #     ai_response = ""
    #     buffer = ""
    #     # 检查 response 是否支持异步迭代
    #     for chunk in response:  # 可能需要用 response.choices 或其他方式
    #         if chunk.choices[0].delta.content is not None:
    #             # 获取流数据中的文本
    #             delta = chunk.choices[0].delta.content
    #             ai_response += delta
    #             print(delta, end="", flush=True)

    #             # 检查是否为完整句子
    #             if re.search(r'[。！？.!?]', buffer):  # 中文和英文标点符号
    #                 text_to_speech(buffer.strip())  # 立即生成并播放语音
    #                 buffer = ""  # 清空缓冲区

    #     # 处理剩余未输出的内容
    #     if buffer:
    #         text_to_speech(buffer.strip())
    #     # 更新历史记录
    #     self.add_to_history(prompt, ai_response)
    #     return ai_response


    def get_openai_response(self, prompt, output_queue):
        self.chat_history.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用正确的模型名称
            messages=self.chat_history,
            stream=True,  # 开启流模式
            temperature=0.7,
        )

        ai_response = ""
        current_sentence = ""

        for chunk in response:
            if 'choices' in chunk and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.get('content', '')
                ai_response += delta
                current_sentence += delta

                # 句子断句逻辑，这里以句号为例
                if current_sentence.endswith(('。', '.', '!', '?', '！', '？')):
                    output_queue.put(current_sentence.strip())  # 放入队列
                    current_sentence = ""  # 清空当前句子

        # 如果最后一句话没有结束符，可以在这里处理
        if current_sentence:
            output_queue.put(current_sentence.strip())

        # 添加完整的历史记录
        self.add_to_history(prompt, ai_response)


    # async def get_openai_response(self, prompt, text_to_speech):
    #     self.chat_history.append({"role": "user", "content": prompt})
        
    #     response = openai.chat.completions.create(
    #         model="gpt-3.5-turbo",  # 确保模型名称正确
    #         messages=self.chat_history,
    #         stream=True,  # 启用流模式
    #         temperature=0.7,
    #     )

    #     ai_response = ""
    #     buffer = ""
    #     async for chunk in response:
    #         if chunk.choices[0].delta.get("content"):
    #             delta = chunk.choices[0].delta["content"]
    #             buffer += delta
    #             ai_response += delta
    #             print(delta, end="", flush=True)

    #             # 检查是否为完整句子
    #             if re.search(r'[。！？.!?]', buffer):  # 中文和英文标点符号
    #                 await text_to_speech(buffer.strip())  # 立即生成并播放语音
    #                 buffer = ""  # 清空缓冲区

    #     # 处理剩余未输出的内容
    #     if buffer:
    #         await text_to_speech(buffer.strip())

    #     # 更新历史记录
    #     self.add_to_history(prompt, ai_response)
    #     return ai_response

    def add_to_history(self, user_input, ai_response):
        self.chat_history.append({"role": "assistant", "content": ai_response})
