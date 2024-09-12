# vad_tool.py

import torch
import torchaudio
import pyaudio
import numpy as np
from rich.console import Console
from df.enhance import enhance, init_df
import soundfile as sf
import openai
import asyncio
import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

console = Console()


# 设置 OpenAI API 密钥
openai.api_key = 'xxxxxx'
openai.base_url="http://localhost:1234/v1/"


import requests
from pydub import AudioSegment
from pydub.playback import play
import asyncio
from io import BytesIO
from edge_tts import Communicate
import audiosegment

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

# 函数：调用 OpenAI API 获取模型回复
def get_openai_response(prompt):
    response = openai.chat.completions.create(
        model="reflection-llama-3.1-8b-solshine-trainround3-16bit-q4_k_m",  # 或者是你使用的模型名称
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
                text_to_speech(current_sentence)  
                collected_messages.append(current_sentence) 
                current_sentence = ""  # 清空当前句子，准备下一个

    if current_sentence != "":
        text_to_speech(current_sentence)  
        collected_messages.append(current_sentence) 
    #使用filter()函数，删除列表中的None值
    total_message =  "".join(list(filter(None, collected_messages)))

    print("")
    # print(total_message)
    return total_message


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Mainly taken from https://github.com/snakers4/silero-vad
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer = []

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return None

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                # end of speak
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer
                self.buffer = []
                return spoken_utterance

        if self.triggered:
            self.buffer.append(x)

        return None


def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound



class VADHandler:
    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
    ):
        """
        Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
        to the following part.
        """

        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load(repo_or_dir='/home/gyc/.cache/torch/hub/snakers4_silero-vad_master', model='silero_vad', trust_repo = True, force_reload = True, source = "local")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            self.enhanced_model, self.df_state, _ = init_df()
            # Enhance model and df state initialization for audio enhancement (currently not implemented)
            pass

    def process(self, audio_chunk):
        """
        Process the audio chunk using VADIterator.

        Parameters
        ----------
        audio_chunk: bytes
            Audio data in raw format.
        """

        if not isinstance(audio_chunk, bytes):
            raise TypeError("Audio chunk must be a bytes object.")

        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"audio input of duration: {len(array) / self.sample_rate}s, skipping"
                )
            else:
                logger.debug("Stop listening")
                if self.audio_enhancement:
                    if self.sample_rate != self.df_state.sr():
                        audio_float32 = torchaudio.functional.resample(
                            torch.from_numpy(array),
                            orig_freq=self.sample_rate,
                            new_freq=self.df_state.sr(),
                        )
                        enhanced = enhance(
                            self.enhanced_model,
                            self.df_state,
                            audio_float32.unsqueeze(0),
                        )
                        enhanced = torchaudio.functional.resample(
                            enhanced,
                            orig_freq=self.df_state.sr(),
                            new_freq=self.sample_rate,
                        )
                    else:
                        enhanced = enhance(
                            self.enhanced_model, self.df_state, audio_float32
                        )
                    array = enhanced.numpy().squeeze()
                yield array


def send_inference_request(file_path, temperature=0.0, temperature_inc=0.2, response_format='json'):  
    """  
    发送推断请求到本地服务器。  
  
    参数:  
    - file_path: 文件的路径，例如 'output.wav'  
    - temperature: 温度参数，默认为 0.0  
    - temperature_inc: 温度增量参数，默认为 0.2  
    - response_format: 响应格式，默认为 'json'  
  
    返回:  
    - 响应的 JSON 数据（如果 response_format 为 'json'）或原始响应内容  
    """  
    url = 'http://127.0.0.1:8080/inference'  
    files = {'file': open(file_path, 'rb')}  # 使用 'rb' 模式打开文件以二进制形式读取  
    data = {  
        'temperature': str(temperature),  
        'temperature_inc': str(temperature_inc),  
        'response_format': response_format  
    }  
      
    headers = {'Content-Type': 'multipart/form-data'}  
      
    # 注意：在 requests 中，当使用 files 参数时，Content-Type 会被自动设置为 multipart/form-data  
    # 因此，手动设置 headers 中的 Content-Type 可能不是必需的，这里为了清晰说明而保留  
      
    response = requests.post(url, files=files, data=data)  
      
    # 检查响应格式，并返回相应的数据  
    if response.status_code == 200:  
        if response_format == 'json':  
            return response.json()  
        else:  
            return response.text  
    else:  
        return response.status_code, response.text  
  

def main():
    should_listen = True
    vad_handler = VADHandler()
    vad_handler.setup(should_listen,audio_enhancement=True)
    
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
                    
    continue_recording = True
    while continue_recording:
        audio_chunk = stream.read(CHUNK)
        
        for output in vad_handler.process(audio_chunk):
            if output is not None:
                sf.write("output.wav", output, SAMPLE_RATE)
                user_text = send_inference_request("/home/gyc/Software/LLaMA/Will-talk-LLM/output.wav")
                print(user_text['text'])
                print("AI:")
                stream.stop_stream()
                response = get_openai_response(user_text['text'])
                stream.start_stream()
                print("")
                print("USER:")
            pass
        
    stream.stop_stream()
    stream.close()
    audio.terminate()


if __name__ == "__main__":
   main()