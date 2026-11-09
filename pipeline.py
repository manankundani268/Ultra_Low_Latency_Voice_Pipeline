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

# ---------- settings (defined once) ----------
max_chunks = 470      # 470 chunks x 32 ms = about 15 s maximum per question
N = 25                # silent chunks needed to end the question (25 x 32 = 800 ms)
speech_prob = 0.5     # a chunk counts as speech at or above this VAD probability

# barge-in rule: speech AND loud, 5 chunks in a row (5 x 32 ms = 160 ms)
PROB_MIN = 0.9
PEAK_MIN = 0.15
NEEDED = 5
PREROLL_CHUNKS = 16   # last 16 chunks (about 0.5 s) kept for later


# ---------- the guard that listens while the assistant talks ----------
def monitor(stream, stop_event, barge_event, preroll):
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
            barge_event.set()      # ring the bell
            return                 # guard goes home


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
    # strip markdown symbols and line breaks, so the voice reads plain text
    for ch in '*#_`':
        s = s.replace(ch, '')
    return s.replace('\n', ' ')


# ---------- speaking that can be interrupted ----------
def is_playing():
    # True while the speaker is still playing something
    try:
        return sd.get_stream().active
    except RuntimeError:       # nothing has been played yet
        return False


def wait_playing(barge_event):
    # wait until the speaker is done OR the bell rings (check every 10 ms)
    while is_playing() and not barge_event.is_set():
        time.sleep(0.01)


def speak(sentence, barge_event):
    with wave.open('reply_part.wav', 'wb') as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    data, rate = sf.read('reply_part.wav')
    wait_playing(barge_event)      # let the previous sentence finish first
    if barge_event.is_set():
        return                     # interrupted: do not start a new sentence
    sd.play(data, rate)            # starts playing and returns immediately


# ---------- record one question from an already-open stream ----------
def record_utterance(stream, start_chunks=None):
    vad = SileroVAD(16000)    # fresh VAD for every question
    speech_started = False
    silent_count = 0
    chunks = []

    if start_chunks is None:
        # normal case: throw away audio that piled up while we were busy
        waiting = stream.read_available
        if waiting > 0:
            stream.read(waiting)
        pending = []
    else:
        # after an interruption: keep it, the saved chunks come first
        pending = list(start_chunks)

    print('Recording started... speak now')
    for i in range(max_chunks):
        if pending:
            chunk = pending.pop(0)            # replay a saved chunk
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

    return chunks, speech_started, silent_count


# ---------- load models once, at startup ----------
start_load = time.perf_counter()
model = WhisperModel('small.en', device='cpu', compute_type='int8')
load_ms = (time.perf_counter() - start_load) * 1000
print(f'Whisper load time: {load_ms:.1f} ms')

start_tts_load = time.perf_counter()
voice = PiperVoice.load("en_US-lessac-medium.onnx")
tts_load_ms = (time.perf_counter() - start_tts_load) * 1000
print(f'Piper load time: {tts_load_ms:.1f} ms')

# warm-up: load the LLM into memory and run the voice once, so the first question isn't slow
start_warm = time.perf_counter()
chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': 'Hi'}],
     options={'num_predict': 1}, keep_alive='30m')
with wave.open('warmup.wav', 'wb') as wav_file:
    voice.synthesize_wav('Hello.', wav_file)
print(f'Warm-up time: {(time.perf_counter() - start_warm) * 1000:.0f} ms')


# ---------- main loop: the mic stays open the whole time ----------
with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:
    
    for _ in range(16):       # discard mic start-up chunks, once
        stream.read(512)

    carry = None
    while True:
        print(f'carry chunks: {0 if carry is None else len(carry)}')
        chunks, speech_started, silent_count = record_utterance(stream,carry
                                                                )
        carry = None
        if not speech_started:
            continue

        print(f'Chunks collected: {len(chunks)}')
        print(f'Recorded duration: {len(chunks) * 32} ms')
        print(f'Silent counts at the end: {silent_count}')

        t_break = time.perf_counter()
        print('Recording finished')

        audio = np.concatenate(chunks)
        audio = audio.flatten()
        sf.write('last_input.wav', audio, 16000)
        print(f'Audio shape: {audio.shape}')

        # transcribe straight from the array, no file
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

        # start the guard BEFORE the assistant starts thinking and talking
        stop_event = threading.Event()
        barge_event = threading.Event()
        preroll = collections.deque(maxlen=PREROLL_CHUNKS)
        t = threading.Thread(target=monitor, args=(stream, stop_event, barge_event, preroll))
        t.start()

        # LLM, speaking sentence by sentence
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
                break                          # bell rang: stop reading the LLM
            piece = part.message.content
            print(piece, end='', flush=True)
            buffer = for_speech(buffer + piece)
            sentences, buffer = split_sentences(buffer)
            for sentence in sentences:
                speak(sentence, barge_event)
                if barge_event.is_set():
                    break                      # bell rang: drop the other sentences
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32

        if not barge_event.is_set():
            leftover = buffer.strip()
            if leftover:
                speak(leftover, barge_event)
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - t_break) * 1000 + N * 32
            wait_playing(barge_event)          # let the last sentence finish

        if barge_event.is_set():
            sd.stop()                          # cut the voice right now

        # send the guard home, and wait until he has left the mic
        stop_event.set()
        t.join()
        if barge_event.is_set():
            print('\nBARGE-IN detected: assistant stopped')
            carry = list(preroll)

        print()
        if first_audio_ms is not None:
            print(f'Voice-to-voice (first audio): {first_audio_ms:.1f} ms')