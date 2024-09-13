import asyncio
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from edge_tts import Communicate

# 函数：使用 edge-tts 将文本转换为语音并播放
def text_to_speech(text):
    # 用于存储音频数据的 BytesIO 对象
    audio_buffer = BytesIO()
    communicate = Communicate(text, voice="zh-CN-XiaoxiaoNeural")  # 使用中文的语音
    
    async def process_stream():
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 处理音频数据
                    audio_data = chunk["data"]
                    # 这里需要根据具体实现播放音频数据，以下是一个示例
                    audio_buffer.write(audio_data)
                # elif chunk["type"] == "WordBoundary":
                #     print(f"WordBoundary: {chunk}")
        except Exception as e:
            print(f"Stream error: {e}")

    try:
        asyncio.run(process_stream())
    finally:
        # 将 BytesIO 对象的指针移动到开头
        audio_buffer.seek(0)
        
        # 从 BytesIO 对象读取 MP3 数据并播放
        audio = AudioSegment.from_mp3(audio_buffer)
        # 播放音频
        play(audio)


def text_to_print(text):
    print("New text:[" + text+"]")
