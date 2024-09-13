import asyncio
from audio_handler import VADHandler
from openai_handler import OpenAIHandler
from tts_handler import text_to_speech
from stt_handler import InferenceHandler
import soundfile as sf
import queue
import threading
from queue import Queue 
import pyttsx3
from threading import Event


output_queue = queue.Queue()
openai_handler = OpenAIHandler()


def speak_from_queue(output_queue,listen_control):
    """
    从队列中读取文本，并用TTS语音合成器进行实时播报。
    
    :param output_queue: 
    """
    if 0:
        engine = pyttsx3.init()  # 初始化TTS引擎
        
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     print("Voice:")
        #     print(" - ID: %s" % voice.id)
        #     print(" - Name: %s" % voice.name)
        #     print(" - Languages: %s" % voice.languages)
        #     print(" - Gender: %s" % voice.gender)
        #     print(" - Age: %s" % voice.age)

        # 设置使用的语音包
        engine.setProperty('voice', 'zh') #开启支持中文
        engine.setProperty('volume', 0.7)
        engine.setProperty('rate', 200)

        while True:
            text = output_queue.get()  # 从队列中取出一个文本
            if not text:  # 如果队列为空，退出循环
                break
            listen_control.clear()
            engine.say(text)  # 将文本转换为语音
            engine.runAndWait()
            listen_control.set()
    else :
        while True:
            text = output_queue.get()  # 从队列中取出一个文本
            if not text:  # 如果队列为空，退出循环
                break
            listen_control.clear()
            text_to_speech(text)
            listen_control.set()


async def process_audio(audio_data):
    inference_handler = InferenceHandler("http://127.0.0.1:8080/inference")
    sf.write("output.wav", audio_data, 16000)
    user_text = inference_handler.send_inference_request("output.wav")
    # Simulate sending inference request and getting user text
    print(user_text)
    if user_text != "":
        print("AI:")
        openai_handler.get_openai_response(user_text, output_queue)
        print("")

        # await text_to_speech(ai_response)
        # pass
    else:
        print("Error Empyt text.")
    
    print("")
    print("USER:")


async def test_process(prompt):
    openai_handler = OpenAIHandler()

    # 启动后台任务来异步处理队列中的文本并进行 TTS
  
    # 启动语音输出线程
    thread = threading.Thread(target=speak_from_queue, args=(output_queue,))
    thread.daemon = True  # 设为守护线程，程序结束时自动销毁线程
    thread.start()  # 启动线程

    while(True):
        openai_handler.get_openai_response(prompt,output_queue)

# # 调用测试函数并传入一个示例的用户输入
# if __name__ == "__main__":
#     asyncio.run(test_process("请帮我生成一些中文句子，用于测试。"))


async def main():
    vad_handler = VADHandler(audio_enhancement=True)
    
    should_listen = Event()
    listen_control = Event()

    should_listen.set()
    listen_control.set()

    # 启动语音输出线程
    thread = threading.Thread(target=speak_from_queue, args=(output_queue,listen_control))
    thread.daemon = True  # 设为守护线程，程序结束时自动销毁线程
    thread.start()  # 启动线程

    print("USER:")
    await vad_handler.record_and_process(listen_control, process_audio)



if __name__ == "__main__":
    asyncio.run(main())
