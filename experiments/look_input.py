import numpy as np
import soundfile as sf
import sounddevice as sd

audio, rate = sf.read('last_input.wav', dtype='float32')
print(f'Length: {len(audio) / rate:.2f} s, peak {np.abs(audio).max():.3f}')

for i in range(0, len(audio) - 511, 512):
    k = np.abs(audio[i:i + 512]).max()          # loudest point in this 32 ms slice
    print(f'{i // 512:3d}  {k:.3f}  ' + '#' * int(k * 40))

sd.play(audio, rate)
sd.wait()