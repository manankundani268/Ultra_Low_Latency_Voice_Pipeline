import wave 
from piper import PiperVoice
import time

start = time.perf_counter()
voice = PiperVoice.load("en_US-lessac-medium.onnx")
end = (time.perf_counter() - start)*1000

start1 = time.perf_counter()
with wave.open('tts_test.wav','wb') as wav_file:
  voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)

with wave.open('tts_test.wav','rb') as wav_file:
  frames = wav_file.getnframes()  #how many samples in file
  rate = wav_file.getframerate()  #samples per seconds 

end1 = (time.perf_counter() - start1)*1000
print(f'Model load time: {end:.1f} ms')
print(f'TTS synthesis time: {end1:.1f} ms')

duration_s = frames/rate

print(rate)
print(frames)
print(duration_s)