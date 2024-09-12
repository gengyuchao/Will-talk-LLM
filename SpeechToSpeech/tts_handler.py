import asyncio
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from edge_tts import Communicate

async def text_to_speech(text):
    audio_buffer = BytesIO()
    communicate = Communicate(text, voice="zh-CN-XiaoxiaoNeural")
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    audio = AudioSegment.from_mp3(audio_buffer)
    play(audio)
