import threading
import collections
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from silero_vad_lite import SileroVAD


PROB_MIN = 0.9          # VAD must be at least this sure
PEAK_MIN = 0.15         # and the mic must be at least this loud
NEEDED = 5              # in a row (5 x 32 ms = 160 ms)
PREROLL_CHUNKS = 16     # keep the last 16 chunks (about 0.5 s) so the start of your speech isn't lost

def monitor(stream,stop_event,barge_event,preroll):
  vad = SileroVAD(16000)
  run = 0
  while not stop_event.is_set():
    chunk,overflowed = stream.read(512)
    preroll.append(chunk)
    p = vad.process(chunk.flatten())
    k = np.abs(chunk).max()
    if p >= PROB_MIN and k >= PEAK_MIN:
      run += 1
    else:
      run = 0

    if run >= NEEDED:
      barge_event.set()
      return

data, rate = sf.read('echo_ref.wav', dtype='float32')
stop_event = threading.Event()
barge_event = threading.Event()
preroll = collections.deque(maxlen=PREROLL_CHUNKS)


with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:
    for _ in range(16):
        stream.read(512)              # discard mic start-up chunks

    t = threading.Thread(target=monitor, args=(stream, stop_event, barge_event, preroll))
    sd.play(data, rate)
    t0 = time.perf_counter()
    t.start()

    # wait until you interrupt, or until the reply is over
    triggered = barge_event.wait(timeout=len(data) / rate + 0.3)
    stop_event.set()
    t.join()
    sd.stop()

elapsed = (time.perf_counter() - t0) * 1000
print(f'Triggered: {triggered}')
print(f'Time since playback start: {elapsed:.0f} ms')
print(f'Pre-roll chunks saved: {len(preroll)}')