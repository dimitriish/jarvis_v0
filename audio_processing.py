import wave
import pyaudio
import numpy as np
import time
import sounddevice as sd
import soundfile as sf

def is_silent(data, threshold=500):
    data = np.frombuffer(data, dtype=np.int16)
    return np.abs(data).mean() < threshold

def record_audio_vad(filename, duration=30, fs=16000, silence_threshold=500, silence_duration=2.0):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    
    print("Recording...")
    frames = []
    silence_start_time = None
    
    while True:
        data = stream.read(1024)
        frames.append(data)
        if is_silent(data, threshold=silence_threshold):
            if silence_start_time is None:
                silence_start_time = time.time()
            elif time.time() - silence_start_time > silence_duration:
                break
        else:
            silence_start_time = None
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

def play_audio(path, fs=16000):
    data, samplerate = sf.read(path)
    if fs is None:
        fs = samplerate

    print("Playing audio...")
    
    # Проигрываем аудио
    sd.play(data, samplerate=fs)
    sd.wait()