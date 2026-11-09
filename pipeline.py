import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
from ollama import chat
import wave
import soundfile as sf
from piper import PiperVoice
from silero_vad_lite import SileroVAD

#Rung-3 splitting the sentences to reduce the latency
def split_sentences(text):
  sentences = []
  start = 0
  for i,ch in enumerate(text):
     if ch in '.?!' and i+1 < len(text) and text[i+1] == ' ':
        word = text[start:i].split(' ')[-1]
        if '.' in word or (len(word) == 1 and word.isupper()):
           continue
        sentences.append(text[start:i+1].strip())
        start = i+1
  leftover = text[start:].strip()
  return sentences, leftover

def for_speech(s):
    # strip markdown symbols and line breaks, so the voice reads plain text
    for ch in '*#_`':
        s = s.replace(ch, '')
    return s.replace('\n', ' ')

def speak(sentence):
    with wave.open('reply_part.wav', 'wb') as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    data, rate = sf.read('reply_part.wav')
    sd.wait()                 # let the previous sentence finish first
    sd.play(data, rate)       # starts playing and returns immediately

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

  #Connecting it to LLM, speaking sentence by sentence
  instruction = 'Answer in three or four short sentences. Each sentence must be under 15 words.'
  llm_stream = chat(
    model='llama3.2:1b',
    messages=[{'role': 'system', 'content': instruction},
              {'role': 'user', 'content': text}],
    stream=True,
  )

  buffer = ''
  first_audio_ms = None
  for part in llm_stream:
    piece = part.message.content
    print(piece, end='', flush=True)
    buffer = for_speech(buffer + piece)
    sentences, buffer = split_sentences(buffer)
    for sentence in sentences:
      speak(sentence)
      if first_audio_ms is None:
        first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32

  leftover = buffer.strip()
  if leftover:
    speak(leftover)
    if first_audio_ms is None:
      first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32
  sd.wait()

  print()
  if first_audio_ms is not None:
    print(f'Voice-to-voice (first audio): {first_audio_ms:.1f} ms')