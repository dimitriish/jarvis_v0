import sounddevice as sd
import numpy as np
import pvporcupine

def wake_word_detect():
    porcupine = pvporcupine.create(keywords=["jarvis"], access_key="WTCduTAVUUhrAqPjy+dRsG9X6c7//0tTGBijdPJzBA7PGEMBK4DYig==")
    with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
        print("Listening for wake word...")
        while True:
            audio_frame, _ = stream.read(porcupine.frame_length)
            audio_frame = audio_frame.flatten()
            audio_frame = np.array(audio_frame, dtype=np.int16)
            keyword_index = porcupine.process(audio_frame)
            if keyword_index >= 0:
                print("Wake word detected!")
                return