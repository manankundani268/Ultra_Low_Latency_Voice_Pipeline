import os
import sounddevice as sd
import numpy as np
import time
import soundfile as sf

sample_rate = 16000 #16,000 Hz is standard for speech
duration = 5   #keeping 5 sec for now

print('Recording Started now... Start speaking')
start_rec = time.perf_counter()
recording = sd.rec(int(sample_rate*duration),samplerate=sample_rate,channels=1)

sd.wait()
elapsed_ms = (time.perf_counter() - start_rec)*1000
print('Recording finished!')

print('Playing back the recorded audio...')
start_play = time.perf_counter()
sd.play(recording,sample_rate)
sd.wait()
elapsed_ms_play = (time.perf_counter() - start_play)*1000
print('Playback finished')

print(f"Numpy array shape: {recording.shape}")
print(recording.dtype)
print(np.abs(recording).max())

print(f'Recording time: {elapsed_ms:.1f} ms')
print(f'PlayBack time: {elapsed_ms_play:.1f} ms')

sf.write('test.wav',recording,sample_rate)

print(os.getcwd())
print(os.path.getsize('test.wav'))