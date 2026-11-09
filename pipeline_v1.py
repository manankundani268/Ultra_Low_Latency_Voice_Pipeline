import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
from ollama import chat
import wave
import soundfile as sf
from piper import PiperVoice
from silero_vad_lite import SileroVAD

sample_rate = 16000
duration = 5

# load the model once, at startup
#Whisper Model
start_load = time.perf_counter()
model = WhisperModel('small.en', device='cpu', compute_type='int8')
load_ms = (time.perf_counter() - start_load) * 1000
print(f'Whisper load time: {load_ms:.1f} ms')

#Piper Model
start_tts_load = time.perf_counter()
voice = PiperVoice.load("en_US-lessac-medium.onnx")
tts_load_ms = (time.perf_counter() - start_tts_load) * 1000
print(f'Piper load time: {tts_load_ms:.1f} ms')

#SileroVAD model integration

while True:
  #SileroVAD model
  vad = SileroVAD(16000)
  max_chunks = 470
  N = 25

  speech_started = False
  silent_count = 0
  chunks = []
  speech_prob = 0.5

  with sd.InputStream(samplerate=16000,channels=1,blocksize=512) as stream:
    for _ in range(16):
      stream.read(512)
    # record from the mic
    print('Recording started... speak now')
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

  if not speech_started:
    continue
  print(f'Chunks collected: {len(chunks)}')
  print(f'Recorded duration: {len(chunks) * 32} ms')
  print(f'Silent counts at the end: {silent_count}')

  t_break = time.perf_counter()

  print('Recording finished')

  audio = np.concatenate(chunks)
  audio = audio.flatten()

  sf.write('last_input.wav',audio,16000)

  print(f'Audio shape: {audio.shape}')

  # transcribe straight from the array, no file
  start_asr = time.perf_counter()
  segments, info = model.transcribe(audio, beam_size=5,language='en')
  text = ''
  for segment in segments:
      text += segment.text
  asr_ms = (time.perf_counter() - start_asr) * 1000
  
  text = text.strip()
  if not text:
    continue

  device = sd.query_devices(kind='input')
  print(f"Input Device: {device['name']}")
  print(f'Peak level of the audio is: {np.abs(audio).max():.3f}')

  print(f'Heard: {text}')
  print(f'ASR time: {asr_ms:.1f} ms')

  #Connecting it to LLM
  start_llm = time.perf_counter()
  instruction = 'Answer in two short sentences. Each sentence must be under 12 words.'

  stream = chat(
    model='llama3.2:1b',
    messages=[{'role': 'system', 'content': instruction},
              {'role': 'user', 'content': text}],
    stream=True,
  )

  pieces = []
  llm_ttft_ms = None
  first_sentence_ms = None
  for chunk in stream:
    if llm_ttft_ms is None:
      llm_ttft_ms = (time.perf_counter() - start_llm) * 1000
    pieces.append(chunk.message.content)
    if first_sentence_ms is None:
      text_so_far = ''.join(pieces)
      if '.' in text_so_far or '?' in text_so_far or '!' in text_so_far:
        first_sentence_ms = (time.perf_counter() - start_llm)*1000
    
    print(chunk.message.content, end='', flush=True)

  llm_total_ms = (time.perf_counter() - start_llm)*1000

  print()
  reply = ''.join(pieces)
  print(f'Reply length : {len(reply)} characters')

  print(f'first TTFT : {llm_ttft_ms:.1f} ms')
  print(f'Total time : {llm_total_ms:.1f} ms')
  if first_sentence_ms is not None:
    print(f'Time for first sentence: {first_sentence_ms:.1f} ms')
  #Connecting it to tts

  start_tts = time.perf_counter()

  with wave.open('reply.wav','wb') as wav_file:
    voice.synthesize_wav(reply, wav_file)

  tts_ms = (time.perf_counter() - start_tts)*1000
  print(f'TTS synthesis time: {tts_ms:.1f} ms')

  data,rate = sf.read('reply.wav')
  print(f'Rate: {rate} Hz')
  sd.play(data,rate)
  v2v_ms = (time.perf_counter() - t_break)*1000 + N*32
  sd.wait()

  processing_ms = asr_ms + llm_total_ms + tts_ms
  print(f'Total Processing time: {processing_ms:.1f} ms')

  print(f'Voice to Voice latency: {v2v_ms:.1f} ms')