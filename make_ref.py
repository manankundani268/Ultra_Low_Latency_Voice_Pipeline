import wave
from piper import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")
text = ("The sky appears blue due to light scattering. "
        "Shorter blue wavelengths are scattered more than red ones. "
        "That is why we see a blue sky during the day.")

with wave.open('echo_ref.wav', 'wb') as wav_file:
    voice.synthesize_wav(text, wav_file)
print('saved echo_ref.wav')