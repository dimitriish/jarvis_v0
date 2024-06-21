import time
from config import stt_pipeline, tts
from ollama import Client
from audio_processing import record_audio_vad, play_audio
from wake_word import wake_word_detect
import sys
sys.path.append('.')
from oscopilot import FridayAgent, ToolManager, FridayExecutor, FridayPlanner, FridayRetriever
from oscopilot.utils import setup_config, setup_pre_run

client = Client(host='http://localhost:11434')
messages = []

def check_intent_and_execute(transcript):
    args = setup_config()
    if not args.query:
        args.query = transcript['text']
    
    task = setup_pre_run(args)
    agent = FridayAgent(FridayPlanner, FridayRetriever, FridayExecutor, ToolManager, config=args)
    result = agent.run(task=task)
    return result

while True:
    wake_word_detect()
    audio_path = "recorded_audio.wav"
    record_audio_vad(audio_path)
    
    transcript = stt_pipeline(
        audio_path=audio_path,
        chunk_length_s=30,
        stride_length_s=5,
        max_new_tokens=128,
        batch_size=100,
        language="english",
        return_timestamps=False,
    )
    print(transcript['text'])
    messages.append({
        'role': 'user',
        'content': transcript['text'],
    })

    response = client.chat(model='llama3', messages=messages)
    messages.append({
        'role': 'assistant',
        'content': response['message']['content'],
    })
    
    if 'bye' in transcript['text'].lower():
        exit()
    # Проверка намерения и выполнение команды через oscopilot
    if "friday" in transcript['text'].lower():
        print("Start planning task...")
        task_result = check_intent_and_execute(transcript)
        print("Task result:", task_result)
        # you might want to add the task_result to messages or process it further.
    print(response['message']['content'])
    tts_audio = tts.tts_to_file(
        text=response['message']['content'],
        file_path="output.wav",
        speaker_wav="speaker.wav",
        language="en"
    )
    
    play_audio(tts_audio)
