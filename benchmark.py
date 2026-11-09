import time
import wave
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from ollama import chat
from piper import PiperVoice
from silero_vad_lite import SileroVAD
from jiwer import wer
import psutil

N = 25
SPEECH_PROB = 0.5
INSTRUCTION = 'Start with one very short sentence of under 8 words. Then add two or three short sentences, each under 15 words.'

model = WhisperModel('small.en', device='cpu', compute_type='int8')
voice = PiperVoice.load('en_US-lessac-medium.onnx')

def synth(sentence):
    t = time.perf_counter()
    with wave.open('bench_reply.wav', 'wb') as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    return (time.perf_counter() - t) * 1000

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

battery = psutil.sensors_battery()
print(f'Plugged in: {battery.power_plugged if battery else "unknown"}')
print(f'Background CPU use before the run: {psutil.cpu_percent(interval=2):.0f} %')

def endpoint(audio):
    # replay the audio through the VAD, and return the audio up to the moment
    # the live endpointing rule would have stopped (or None if nobody spoke)
    vad = SileroVAD(16000)
    speech_started = False
    silent_count = 0
    for start in range(0, len(audio) - 511, 512):
        prob = vad.process(audio[start:start + 512])
        if prob >= SPEECH_PROB:
            speech_started = True
            silent_count = 0
        else:
            silent_count += 1
        if speech_started and silent_count >= N:
            return audio[:start + 512]
    return None

def run_once(audio):
    trimmed = endpoint(audio)
    if trimmed is None:
        return None

    t = time.perf_counter()
    segments, info = model.transcribe(trimmed, beam_size=5, language='en', temperature=0.0)
    text = ''.join(s.text for s in segments).strip()
    asr_ms = (time.perf_counter() - t) * 1000
    if not text:
        return None

    t = time.perf_counter()
    reply = ''
    buffer = ''
    first_sentence = None
    first_sentence_ms = None
    for piece in chat(model='llama3.2:1b', stream=True, options={'temperature': 0},
                      messages=[{'role': 'system', 'content': INSTRUCTION},
                                {'role': 'user', 'content': text}]):
        reply += piece.message.content
        if first_sentence is None:
            buffer = for_speech(buffer + piece.message.content)
            sentences, buffer = split_sentences(buffer)
            if sentences:
                first_sentence = sentences[0]
                first_sentence_ms = (time.perf_counter() - t) * 1000
    llm_ms = (time.perf_counter() - t) * 1000

    if first_sentence is None:      # one-sentence reply: the whole reply is the first sentence
        first_sentence = for_speech(reply).strip()
        first_sentence_ms = llm_ms

    tts_ms = synth(for_speech(reply))
    tts1_ms = synth(first_sentence)
    return text, asr_ms, llm_ms, tts_ms, first_sentence_ms, tts1_ms

def clean(s):
    s = s.lower()
    for ch in '?.,!':
        s = s.replace(ch, '')
    return s.strip()

REPEATS = 5

with open('testset/questions.txt') as f:
    questions = [line.strip() for line in f if line.strip()]

# warm-up pass on one real question, not recorded
warm_audio, sr = sf.read('testset/q02.wav', dtype='float32')
run_once(warm_audio)

rows = []
references = []
hypotheses = []
for i, question in enumerate(questions, start=1):
    audio, sr = sf.read(f'testset/q{i:02d}.wav', dtype='float32')
    for rep in range(REPEATS):
        result = run_once(audio)
        if result is None:
            break
        text, asr_ms, llm_ms, tts_ms, first_sentence_ms, tts1_ms = result
        seq_total = N * 32 + asr_ms + llm_ms + tts_ms
        stream_total = N * 32 + asr_ms + first_sentence_ms + tts1_ms
        rows.append((asr_ms, llm_ms, tts_ms, seq_total, stream_total))
        if rep == 0:
            references.append(clean(question))
            hypotheses.append(clean(text))
            print(f'q{i:02d}: heard: {text}')

print()
print(f'Timed runs: {len(rows)}')
for name, col in [('ASR', 0), ('LLM', 1), ('TTS', 2),
                  ('Sequential voice-to-voice', 3), ('Streaming first-audio', 4)]:
    values = [r[col] for r in rows]
    print(f'{name}: p50 {np.percentile(values, 50):.0f} ms, p95 {np.percentile(values, 95):.0f} ms')
print(f'Endpointing wait: {N * 32} ms (fixed)')
print(f'Word error rate: {wer(references, hypotheses) * 100:.1f} %')