from faster_whisper import WhisperModel
import time
import soundfile as sf

audio,sr = sf.read('test.wav',dtype='float32')

start = time.perf_counter()

model = WhisperModel('tiny',device='cpu',compute_type='int8')
total = (time.perf_counter() - start)*1000
print(f'Model load time: {total:.1f} ms')

start1 = time.perf_counter()
segments,info = model.transcribe(audio,beam_size=5)

for segment in segments:
  print("[%.2fs -> %.2fs] %s" % (segment.start,segment.end,segment.text))

total1 = (time.perf_counter() - start1)*1000

print(f'Model transcription time: {total1:.1f} ms')

print(audio.shape)
print(sr)