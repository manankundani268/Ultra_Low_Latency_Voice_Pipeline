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
min_speech_chunks = 12       # below this, the sound is treated as noise

PROB_MIN = 0.9
PEAK_MIN = 0.15
NEEDED = 5
PREROLL_CHUNKS = 16

MAX_EXCHANGES = 3            # NEW: remember the last 3 question/answer pairs
history = []                 # NEW: list of {'role', 'content'} messages


# ---------- monitor thread (barge-in guard) ----------
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
            bell_time.append(time.perf_counter())
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
    speech_count = 0
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
            speech_count += 1
        else:
            silent_count += 1

        if speech_started and silent_count >= N:
            break

    return chunks, speech_started, silent_count, speech_count


# ---------- load models ----------
start_load = time.perf_counter()
model = WhisperModel('small.en', device='cpu', compute_type='int8')
print(f'Whisper load time: {(time.perf_counter() - start_load) * 1000:.1f} ms')

start_tts_load = time.perf_counter()
voice = PiperVoice.load("en_US-lessac-medium.onnx")
print(f'Piper load time: {(time.perf_counter() - start_tts_load) * 1000:.1f} ms')

start_warm = time.perf_counter()
chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': 'Hi'}],
     options={'num_predict': 1}, keep_alive='30m')
with wave.open('warmup.wav', 'wb') as wav_file:
    voice.synthesize_wav('Hello.', wav_file)
print(f'Warm-up time: {(time.perf_counter() - start_warm) * 1000:.0f} ms')


# ---------- main loop ----------
try:
    with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:
        for _ in range(16):
            stream.read(512)

        carry = None
        while True:
            chunks, speech_started, silent_count, speech_count = record_utterance(stream, carry)
            carry = None
            if not speech_started:
                continue
            if speech_count < min_speech_chunks:
                print(f'Skipped: only {speech_count} speech chunks (need {min_speech_chunks})')
                continue

            print(f'Recorded duration: {len(chunks) * 32} ms')
            t_break = time.perf_counter()

            audio = np.concatenate(chunks).flatten()
            if np.abs(audio).max() < 0.08:
                continue
            sf.write('last_input.wav', audio, 16000)

            start_asr = time.perf_counter()
            segments, info = model.transcribe(audio, beam_size=5, language='en', temperature=0.0)
            text = ''.join(segment.text for segment in segments).strip()
            asr_ms = (time.perf_counter() - start_asr) * 1000
            if not text:
                continue

            print(f'Heard: {text}')
            print(f'ASR time: {asr_ms:.1f} ms')

            instruction = 'Answer in three or four short sentences. Each sentence must be under 15 words.You remember the users details from earlier in this conversation. Use them when asked.'

            stop_event = threading.Event()
            barge_event = threading.Event()
            preroll = collections.deque(maxlen=PREROLL_CHUNKS)
            bell_time = []
            t = threading.Thread(target=monitor, args=(stream, stop_event, barge_event, preroll, bell_time))
            t.start()

            # NEW: the LLM sees the system instruction, the last exchanges, then the new question
            messages = [{'role': 'system', 'content': instruction}] + history + [{'role': 'user', 'content': text}]
            llm_stream = chat(model='llama3.2:1b', messages=messages, stream=True)

            buffer = ''
            full_reply = ''                      # NEW: collect the whole answer for memory
            first_audio_ms = None
            for part in llm_stream:
                if barge_event.is_set():
                    break
                piece = part.message.content
                print(piece, end='', flush=True)
                full_reply += piece
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

            stop_event.set()
            t.join()

            # NEW: save the exchange only if the answer was finished, then cap the history
            if not barge_event.is_set() and full_reply.strip():
                history.append({'role': 'user', 'content': text})
                history.append({'role': 'assistant', 'content': full_reply.strip()})
                history[:] = history[-2 * MAX_EXCHANGES:]

            if barge_event.is_set():
                print('\nBARGE-IN detected: assistant stopped')
                carry = list(preroll)

            print()
            if first_audio_ms is not None:
                print(f'Voice-to-voice (first audio): {first_audio_ms:.1f} ms')

except KeyboardInterrupt:
    sd.stop()
    print('\nStopped.')