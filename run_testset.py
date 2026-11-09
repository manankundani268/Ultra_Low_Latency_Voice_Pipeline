import time
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer

def clean(s):
    s = s.lower()
    for ch in '?.,!':
        s = s.replace(ch, '')
    return s.strip()

# load the model once
start_load = time.perf_counter()
model = WhisperModel('base', device='cpu', compute_type='int8')
load_ms = (time.perf_counter() - start_load) * 1000
print(f'Model load time: {load_ms:.1f} ms\n')

# read the expected questions
with open('testset/questions.txt') as f:
    questions = [line.strip() for line in f if line.strip()]

references = []
hypotheses = []


for i, question in enumerate(questions, start=1):
    filename = f'testset/q{i:02d}.wav'
    audio, sr = sf.read(filename, dtype='float32')

    # transcribe, timed (the timer stops after the loop over segments)
    start_asr = time.perf_counter()
    segments, info = model.transcribe(audio, beam_size=5, language='en')
    text = ''
    for segment in segments:
        text += segment.text
    asr_ms = (time.perf_counter() - start_asr) * 1000

    print(f'File: {filename}')
    print(f'Expected: {question}')
    print(f'Heard:    {text.strip()}')
    print(f'ASR time: {asr_ms:.1f} ms')
    if i != 1:
      references.append(clean(question))
      hypotheses.append(clean(text))

error_rate = wer(references, hypotheses)
print(f'\nWord error rate over {len(references)} files: {error_rate * 100:.1f} %')