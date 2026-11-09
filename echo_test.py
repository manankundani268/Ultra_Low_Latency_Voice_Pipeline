import numpy as np
import sounddevice as sd
import soundfile as sf
from silero_vad_lite import SileroVAD

data, rate = sf.read('reply_part.wav', dtype='float32')
vad = SileroVAD(16000)

sd.play(data, rate)
n_chunks = int(len(data) / rate * 16000 / 512)
peaks = []
probs = []
with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:
    for i in range(n_chunks):
        chunk, overflowed = stream.read(512)
        peaks.append(np.abs(chunk).max())
        probs.append(vad.process(chunk.flatten()))
sd.wait()

for name, values in [('mic peak', peaks), ('VAD probability', probs)]:
    print(f'{name}: median {np.median(values):.3f}, p90 {np.percentile(values, 90):.3f}, max {max(values):.3f}')
print(f'Chunks with probability >= 0.5: {sum(p >= 0.5 for p in probs)} of {n_chunks}')
print(f'Chunks with probability >= 0.9: {sum(p >= 0.9 for p in probs)} of {n_chunks}')

# NEW: longest streak of consecutive chunks that meet the barge-in rule
run = 0
longest = 0
for p, k in zip(probs, peaks):
    if p >= 0.9 and k >= 0.15:
        run += 1
        longest = max(longest, run)
    else:
        run = 0
print(f'Longest run of chunks meeting the barge-in rule: {longest} (the rule needs 5)')