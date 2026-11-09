import sounddevice as sd
import soundfile as sf
import numpy as np

sample_rate = 16000
duration = 7

with open('testset/questions.txt') as f:
  questions = [line.strip() for line in f if line.strip()]

for i,question in enumerate(questions,start=1):
  print(f'\nQuestion {i}/{len(questions)}: {question}')
  input('Press Enter, wait about 1 second, then say it...')

  recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
  sd.wait()
  audio = recording.flatten()

  filename = f'testset/q{i:02d}.wav'
  sf.write(filename, audio, sample_rate)
  print(f'Saved {filename}, shape {audio.shape}, peak {np.abs(audio).max():.3f}')