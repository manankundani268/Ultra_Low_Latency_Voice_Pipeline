import sounddevice as sd
import numpy as np

max_chunks = 470
threshold = 0.04
N = 25

speech_started = False
silent_count = 0
chunks = []

with sd.InputStream(samplerate=16000,channels=1,blocksize=512) as stream:
  for i in range(max_chunks):
    chunk,overflowed = stream.read(512)
    chunks.append(chunk)
    peak = np.abs(chunk).max()

    if peak >= threshold :
      speech_started = True
      silent_count = 0
    else:
      silent_count+=1

    if speech_started and silent_count>=N:
      break

print(f'Chunks collected: {len(chunks)}')
print(f'Recorded duration: {len(chunks) * 32} ms')
print(f'Silent counts at the end: {silent_count}')