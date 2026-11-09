import time
import statistics
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer

def clean(s):
    s = s.lower()
    for ch in '?.,!':
        s = s.replace(ch, '')
    return s.strip()

with open('testset/questions.txt') as f:
    questions = [line.strip() for line in f if line.strip()]

for name in ['small.en', 'distil-small.en']:
    model = WhisperModel(name, device='cpu', compute_type='int8')

    for beam in [5, 1]:
        references = []
        hypotheses = []
        times = []

        for i, question in enumerate(questions, start=1):
            audio, sr = sf.read(f'testset/q{i:02d}.wav', dtype='float32')

            start_asr = time.perf_counter()
            segments, info = model.transcribe(audio, beam_size=beam, language='en')
            text = ''
            for segment in segments:
                text += segment.text
            asr_ms = (time.perf_counter() - start_asr) * 1000

            if i != 1:
                references.append(clean(question))
                hypotheses.append(clean(text))
                times.append(asr_ms)

        print(f'{name} beam={beam}: WER {wer(references, hypotheses) * 100:.1f} %, '
              f'median ASR {statistics.median(times):.0f} ms, '
              f'max ASR {max(times):.0f} ms')