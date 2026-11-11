import sounddevice as sd
import numpy as np
from silero_vad_lite import SileroVAD

n_chunks = 94

vad = SileroVAD(16000)

with sd.InputStream(samplerate=16000, channels=1, blocksize=512) as stream:
    for i in range(n_chunks):
        chunk, overflowed = stream.read(512)
        peak = np.abs(chunk).max()
        prob = vad.process(chunk.flatten())
        print(f'Chunk {i}: peak {peak:.3f}, speech probability {prob:.3f}')