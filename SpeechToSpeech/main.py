import asyncio
from audio_handler import VADHandler
from openai_handler import OpenAIHandler
from tts_handler import text_to_speech
from stt_handler import InferenceHandler
import soundfile as sf
import queue

async def process_queue(output_queue):
    while True:
        sentence = await asyncio.get_event_loop().run_in_executor(None, output_queue.get)
        if sentence == "STOP":
            break
        await text_to_speech(sentence)  # 异步调用 TTS

async def process_audio(audio_data):
    inference_handler = InferenceHandler("http://127.0.0.1:8080/inference")
    sf.write("output.wav", audio_data, 16000)
    user_text =  await inference_handler.send_inference_request("output.wav")
    # Simulate sending inference request and getting user text
    # user_text = "这是模拟的推断结果。"  # Replace with actual inference call
    print("USER:", user_text)
    if user_text != "":
        openai_handler = OpenAIHandler()
        output_queue = queue.Queue()

        # 启动后台任务来异步处理队列中的文本并进行 TTS
        tts_task = asyncio.create_task(process_queue(output_queue))

        print("AI:")
 
        # 使用 ThreadPoolExecutor 在异步环境中运行同步的 get_openai_response
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, openai_handler.get_openai_response, user_text, output_queue)

        # 发送停止信号以结束 TTS 处理任务
        output_queue.put("STOP")
        await tts_task

        # await text_to_speech(ai_response)
        # pass
    else:
        print("Error Empyt text.")


async def test_process(prompt):
    openai_handler = OpenAIHandler()
    output_queue = queue.Queue()

    # 启动后台任务来异步处理队列中的文本并进行 TTS
    tts_task = asyncio.create_task(process_queue(output_queue))

    # # 使用 ThreadPoolExecutor 在异步环境中运行同步的 get_openai_response
    # loop = asyncio.get_running_loop()
    # with ThreadPoolExecutor() as pool:
    #     await loop.run_in_executor(pool, openai_handler.get_openai_response, prompt, output_queue)

    # # 等待 OpenAI 响应完全处理完再发送停止信号
    # output_queue.put("STOP")
    # await tts_task

    openai_handler.get_openai_response(prompt)

# 调用测试函数并传入一个示例的用户输入
if __name__ == "__main__":
    asyncio.run(test_process("请帮我生成一些中文句子，用于测试。"))


# async def main():
#     # vad_handler = VADHandler(audio_enhancement=True)
#     # should_listen = True
#     # await vad_handler.record_and_process(should_listen, process_audio)

#     await test_process()


# if __name__ == "__main__":
#     asyncio.run(main())
