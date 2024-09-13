import asyncio
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from edge_tts import Communicate


import pyaudio
import edge_tts

# 配置 Edge TTS
VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_FILE = "output.mp3"

# 配置 PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000


async def stream_tts(text):
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)

    # 创建 Edge TTS 的流式生成器
    communicate = edge_tts.Communicate(text, VOICE)

    # 逐块读取并播放音频数据
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            stream.write(chunk["data"])

    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

async def get_audio_from_communicate(text):
    audio_buffer = BytesIO()
    communicate = Communicate(text, voice="zh-CN-XiaoxiaoNeural")  # 使用中文的语音
    async def stream():
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

    await stream()
    
    # 将 BytesIO 对象的指针移动到开头
    audio_buffer.seek(0)
    
    return audio_buffer


async def play_audio(audio_buffer):
    # 从 BytesIO 对象读取 MP3 数据并播放
    audio = AudioSegment.from_mp3(audio_buffer)
    # 播放音频
    play(audio)

async def text_to_speech(text):
    audio_buffer = BytesIO()
    communicate = Communicate(text, voice="zh-CN-XiaoxiaoNeural")  # 使用中文的语音
    async def stream():
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

    await stream()



# 函数：使用 edge-tts 将文本转换为语音并播放
def text_to_speech_orig(text):
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
