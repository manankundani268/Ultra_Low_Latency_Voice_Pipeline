import time
import soundfile as sf
from faster_whisper import WhisperModel

audio, sr = sf.read('last_input.wav', dtype='float32')
print(f'Audio shape: {audio.shape}, rate: {sr} Hz')

sizes = ['tiny', 'base', 'small']

for size in sizes:
    # load the model, timed on its own
    start_load = time.perf_counter()
    model = WhisperModel(size, device='cpu', compute_type='int8')
    load_ms = (time.perf_counter() - start_load) * 1000

    # transcribe, timed on its own (the timer stops after the loop)
    start_asr = time.perf_counter()
    segments, info = model.transcribe(audio, beam_size=5, language='en')
    text = ''
    for segment in segments:
        text += segment.text
    asr_ms = (time.perf_counter() - start_asr) * 1000

    print(f'Model: {size}')
    print(f'Load time: {load_ms:.1f} ms')
    print(f'ASR time: {asr_ms:.1f} ms')
    print(f'Heard: {text.strip()}')