import sounddevice as sd
import numpy as np
from silero_vad_lite import SileroVAD

max_chunks = 470
N = 25

speech_started = False
silent_count = 0
chunks = []
speech_prob = 0.5

vad = SileroVAD(16000)


with sd.InputStream(samplerate=16000,channels=1,blocksize=512) as stream:
  for i in range(max_chunks):
    chunk,overflowed = stream.read(512)
    chunks.append(chunk)
    prob = vad.process(chunk.flatten())

    if prob >= speech_prob :
      speech_started = True
      silent_count = 0
    else:
      silent_count+=1

    if speech_started and silent_count>=N:
      break

print(f'Chunks collected: {len(chunks)}')
print(f'Recorded duration: {len(chunks) * 32} ms')
print(f'Silent counts at the end: {silent_count}')