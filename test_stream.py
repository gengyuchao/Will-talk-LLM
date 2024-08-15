# -*- coding: utf-8 -*-

# 退格键出现问题，导入readline库即可

import openai
import edge_tts
import asyncio
from pydub import AudioSegment
from pydub.playback import play
import io
import os
import sys
import readline
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'

sys.stdin.reconfigure(encoding='utf-8')

client_tts = openai.OpenAI(base_url="http://127.0.0.1:6006/v1", api_key="xxxxxx")

# 设置 OpenAI API 密钥
openai.api_key = 'xxxxxx'
openai.base_url="http://localhost:1234/v1/"

# 函数：调用 OpenAI API 获取模型回复
async def get_openai_response(prompt):
    response = openai.chat.completions.create(
        model="lmstudio-community/xxxxxx",  # 或者是你使用的模型名称
        messages=[
            # {"role": "system", "content": "You will follow the steps below: \n1. Translate the question into English and display it. \n2. Answer questions in English. \n3.Translate the answer to chinese.\n4. Stop Generation."},
            {"role": "system", "content": "你是一个非常聪明的人工智能，请自然的回答用户提出的各种问题。"},
            {"role": "user", "content": prompt},
        ],
        stream=True,  # 启用流模式
        temperature=0.7,
    )

    collected_messages = []
    current_sentence = ""
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content 
        if chunk_message != None:
            current_sentence += chunk_message  # 将片段累积到当前句子中

            print(chunk_message, end="", flush=True)

            if chunk_message.endswith(".") or chunk_message.endswith("\n") or chunk_message.endswith("。") or chunk_message.find(":") != -1 or chunk_message.endswith("：") : # 判断是否是一个完整的句子
                await text_to_speech_local(current_sentence)  
                collected_messages.append(current_sentence) 
                current_sentence = ""  # 清空当前句子，准备下一个

    #使用filter()函数，删除列表中的None值
    total_message =  "".join(list(filter(None, collected_messages)))

    print("")
    # print(total_message)
    return total_message

# 函数：使用 edge-tts 将文本转换为语音并播放
async def text_to_speech(text):
    # 用于存储音频数据的 BytesIO 对象
    audio_buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")  # 使用中文的语音
    with open('output.wav', 'wb') as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                # 处理音频数据
                audio_data = chunk["data"]
                # 这里需要根据具体实现播放音频数据，以下是一个示例
                f.write(audio_data)
                audio_buffer.write(audio_data)
            # elif chunk["type"] == "WordBoundary":
            #     print(f"WordBoundary: {chunk}")
    
    # 将 BytesIO 对象的指针移动到开头
    audio_buffer.seek(0)
    
    # 从 BytesIO 对象读取 MP3 数据并播放
    audio = AudioSegment.from_mp3(audio_buffer)
    # 播放音频
    play(audio)

# 函数：使用 edge-tts 将文本转换为语音并播放
async def text_to_speech_local(text):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client_tts.audio.speech.create(
        model="tts-1",
        voice="7495", # 5480  8051  
        input=text
    )
    response.stream_to_file(speech_file_path)

    # 从 BytesIO 对象读取 MP3 数据并播放
    audio = AudioSegment.from_mp3("speech.mp3")
    # 播放音频
    play(audio)

# 主程序
async def main():
    user_input = input("请输入你的问题: ")
    # print("你输入的是："+user_input)
    response = await get_openai_response(user_input)

    print(response)
    
    # 调用异步的 TTS 函数
    # await text_to_speech(response)
    # await text_to_speech_local(response)

if __name__ == "__main__":
    while(True):
        asyncio.run(main())

