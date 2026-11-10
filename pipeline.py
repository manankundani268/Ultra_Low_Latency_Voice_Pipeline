# ---------- imports ----------
import sounddevice as sd
import numpy as np
import time
import threading
import collections
from faster_whisper import WhisperModel
from ollama import chat
import wave
import soundfile as sf
from piper import PiperVoice
from silero_vad_lite import SileroVAD

# ---------- settings ----------
max_chunks = 470
N = 25
speech_prob = 0.5

PROB_MIN = 0.9
PEAK_MIN = 0.15
NEEDED = 5
PREROLL_CHUNKS = 16


# ---------- monitor thread ----------
def monitor(stream, stop_event, barge_event, preroll, bell_time):  
    vad = SileroVAD(16000)
    run = 0
    while not stop_event.is_set():
        chunk, overflowed = stream.read(512)
        preroll.append(chunk)
        p = vad.process(chunk.flatten())
        k = np.abs(chunk).max()
        if p >= PROB_MIN and k >= PEAK_MIN:
            run += 1
        else:
            run = 0
        if run >= NEEDED:
            bell_time.append(time.perf_counter())  # NEW: record when bell rang
            barge_event.set()
            return


# ---------- sentence splitting ----------
def split_sentences(text):
    sentences = []
    start = 0
    for i, ch in enumerate(text):
        if ch in '.?!' and i + 1 < len(text) and text[i + 1] == ' ':
            word = text[start:i].split(' ')[-1]
            if '.' in word or (len(word) == 1 and word.isupper()):
                continue
            sentences.append(text[start:i + 1].strip())
            start = i + 1
    leftover = text[start:].strip()
    return sentences, leftover


def for_speech(s):
    for ch in '*#_`':
        s = s.replace(ch, '')
    return s.replace('\n', ' ')


# ---------- speaking that can be interrupted ----------
def is_playing():
    try:
        return sd.get_stream().active
    except RuntimeError:
        return False


def wait_playing(barge_event):
    while is_playing() and not barge_event.is_set():
        time.sleep(0.01)


def speak(sentence, barge_event):
    with wave.open('reply_part.wav', 'wb') as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    data, rate = sf.read('reply_part.wav')
    wait_playing(barge_event)
    if barge_event.is_set():
        return
    sd.play(data, rate)


# ---------- record one question ----------
def record_utterance(stream, start_chunks=None):
    vad = SileroVAD(16000)
    speech_started = False
    silent_count = 0
    chunks = []

    if start_chunks is None:
        waiting = stream.read_available
        if waiting > 0:
            stream.read(waiting)
        pending = []
    else:
        pending = list(start_chunks)

    print('Recording started... speak now')
    for i in range(max_chunks):
        if pending:
            chunk = pending.pop(0)
        else:
            chunk, overflowed = stream.read(512)
        chunks.append(chunk)
        prob = vad.process(chunk.flatten())

        if prob >= speech_prob:
            speech_started = True
            silent_count = 0
        else:
            silent_count += 1

        if speech_started and silent_count >= N:
            break

    speech_count = sum(1 for prob in [vad.process(c.flatten()) for c in chunks] if prob >= 0.5)
    return chunks, speech_started, silent_count, speech_count


# ---------- load models ----------
start_load = time.perf_counter()
model = WhisperModel('small.en', device='cpu', compute_type='int8')
load_ms = (time.perf_counter() - start_load) * 1000
print(f'Whisper load time: {load_ms:.1f} ms')

start_tts_load = time.perf_counter()
voice = PiperVoice.load("en_US-lessac-medium.onnx")
tts_load_ms = (time.perf_counter() - start_tts_load) * 1000
print(f'Piper load time: {tts_load_ms:.1f} ms')

start_warm = time.perf_counter()
chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': 'Hi'}],
     options={'num_predict': 1}, keep_alive='30m')
with wave.open('warmup.wav', 'wb') as wav_file:
    voice.synthesize_wav('Hello.', wav_file)
print(f'Warm-up time: {(time.perf_counter() - start_warm) * 1000:.0f} ms')


# ---------- main loop ----------
try:                                                   # NEW: catch Ctrl+C cleanly
    with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:

        for _ in range(16):
            stream.read(512)

        carry = None
        while True:
            print(f'carry chunks: {0 if carry is None else len(carry)}')
            chunks, speech_started, silent_count,speech_count = record_utterance(stream, carry)
            carry = None
            if not speech_started:
                continue

            if speech_count < 8:
                continue

            print(f'Chunks collected: {len(chunks)}')
            print(f'Recorded duration: {len(chunks) * 32} ms')
            print(f'Silent counts at the end: {silent_count}')

            t_break = time.perf_counter()
            print('Recording finished')

            audio = np.concatenate(chunks)
            audio = audio.flatten()
            if np.abs(audio).max() < 0.08:
                continue
            sf.write('last_input.wav', audio, 16000)
            print(f'Audio shape: {audio.shape}')

            start_asr = time.perf_counter()
            segments, info = model.transcribe(audio, beam_size=5, language='en', temperature=0.0)
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

            instruction = 'Answer in three or four short sentences. Each sentence must be under 15 words.'

            stop_event = threading.Event()
            barge_event = threading.Event()
            preroll = collections.deque(maxlen=PREROLL_CHUNKS)
            bell_time = []                             # NEW: empty list for the bell time
            t = threading.Thread(
                target=monitor,
                args=(stream, stop_event, barge_event, preroll, bell_time)  # NEW: pass bell_time
            )
            t.start()

            llm_stream = chat(
                model='llama3.2:1b',
                messages=[{'role': 'system', 'content': instruction},
                          {'role': 'user', 'content': text}],
                stream=True,
            )

            buffer = ''
            first_audio_ms = None
            for part in llm_stream:
                if barge_event.is_set():
                    break
                piece = part.message.content
                print(piece, end='', flush=True)
                buffer = for_speech(buffer + piece)
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    speak(sentence, barge_event)
                    if barge_event.is_set():
                        break
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32

            if not barge_event.is_set():
                leftover = buffer.strip()
                if leftover:
                    speak(leftover, barge_event)
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32
                wait_playing(barge_event)

            if barge_event.is_set():
                sd.stop()
                if bell_time:                          # NEW: reaction time
                    reaction_ms = (time.perf_counter() - bell_time[0]) * 1000
                    print(f'\nBarge-in reaction time: {reaction_ms:.0f} ms')

            stop_event.set()
            t.join()
            if barge_event.is_set():
                print('BARGE-IN detected: assistant stopped')
                carry = list(preroll)

            print()
            if first_audio_ms is not None:
                print(f'Voice-to-voice (first audio): {first_audio_ms:.1f} ms')

except KeyboardInterrupt:                              # NEW: clean exit
    sd.stop()
    print('\nStopped.')