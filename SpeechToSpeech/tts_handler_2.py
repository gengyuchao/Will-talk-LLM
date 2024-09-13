import queue
import threading
import edge_tts  # 假设你已经安装了 edge_tts 库
import pyaudio    # 用于播放音频
import wave
from pydub import AudioSegment
from pydub.playback import play

from io import BytesIO
# 创建两个 Queue
sentence_queue = queue.Queue()
audio_queue = queue.Queue()

# 创建一个 Event 用于控制任务是否结束
stop_event = threading.Event()

VOICE = "zh-CN-XiaoxiaoNeural"

# 任务1：从句子队列中获取句子，生成音频并放入音频队列
def generate_audio():
    mp3_count = 0
    while not stop_event.is_set():
        try:
            sentence = sentence_queue.get(timeout=1)  # 从句子队列中获取句子，超时时间为1秒
            if sentence is None:  # 如果接收到 None，表示任务结束
                break
            # 使用 edge_tts 生成音频
            communicate = edge_tts.Communicate(sentence, VOICE)
            file_name = "voice_history/v" + str(mp3_count) + ".mp3"
            communicate.save_sync(file_name)
            audio_queue.put(file_name)  # 将音频放入音频队列
            mp3_count = mp3_count + 1
            # print(f"mp3_count:{mp3_count}")
        except queue.Empty:
            continue  # 如果没有任务，继续等待


# 任务2：从音频队列中获取音频并播放
def play_audio(listen_control,generating):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True)

    while not stop_event.is_set():
        try:
            audio_path = audio_queue.get(timeout=2)  # 从音频队列中获取音频，超时时间为1秒
            if audio_path is None:  # 如果接收到 None，表示任务结束
                break
            
            # 读取临时文件中的音频数据

            # 将 MP3 文件转换为 WAV 文件
            sound = AudioSegment.from_mp3(audio_path)

            # 播放音频
            play(sound)
            
            if audio_queue.empty() and sentence_queue.empty() and (not generating.is_set()):
                listen_control.set()
                print("listen_control.set()")
        
        except queue.Empty:
            continue  # 如果没有任务，继续等待
    
    # 关闭流和 PyAudio 实例
    stream.stop_stream()
    stream.close()
    p.terminate()
    
# # 启动两个线程
# threading.Thread(target=generate_audio).start()
# threading.Thread(target=play_audio).start()

# # 向句子队列中添加句子
# sentences = ["Hello, how are you?", "This is a test.", "Python is awesome!"]
# for sentence in sentences:
#     sentence_queue.put(sentence)

# # 等待一段时间，模拟任务进行
# import time
# time.sleep(10)

# # 设置停止事件，通知线程任务结束
# stop_event.set()

# # 向队列中放入 None，通知线程任务结束
# sentence_queue.put(None)
# audio_queue.put(None)