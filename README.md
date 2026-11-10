# Real-Time Voice-to-Voice Assistant (CPU)

A voice assistant that runs entirely on a laptop CPU: it listens, transcribes, thinks, and speaks back. It is built from scratch in Python, and every stage is timed so the bottlenecks are measured, not guessed.

**Headline result:** streaming first audio **3.3 s (p50) / 3.6 s (p95)** after the user stops talking, with `llama3.2:3b`, on a laptop CPU, plugged in. Word error rate **16.7%** on a 9-question test set.

---

## Features

- **Endpointing:** Silero VAD decides when you have stopped talking (800 ms of silence).
- **Sentence streaming:** the reply is spoken sentence by sentence while the LLM is still writing.
- **Barge-in:** speak over the assistant and it stops, then answers your new question. Your interrupting speech is kept (pre-roll), so the start of it is not lost.
- **Noise gate:** short noises and clicks are ignored before transcription.
- **Short-term memory:** the last 3 exchanges are kept in the prompt (not saved to disk).

## Architecture

```
Mic → Silero VAD → Whisper (small.en) → Llama 3.2 3B (Ollama) → Piper TTS → Speaker
        ▲                                                             │
        └──────── barge-in monitor (runs while the assistant talks) ──┘
```

| Stage | Tool | Role |
|---|---|---|
| Capture / playback | `sounddevice` | 16 kHz mic input, 32 ms chunks |
| Turn detection | Silero VAD (`silero-vad-lite`) | Speech probability per chunk, end-of-turn rule |
| Speech recognition | `faster-whisper` `small.en` (int8, CPU) | Speech to text |
| Response generation | Ollama, `llama3.2:3b` | Streamed token by token |
| Speech synthesis | Piper `en_US-lessac-medium` | Speaks each finished sentence |

## Results

All numbers are from one laptop, on AC power, Windows "Best performance" mode. The benchmark replays 9 recorded questions, 5 times each (45 timed runs).

| Stage | p50 | p95 |
|---|---|---|
| Endpointing wait (fixed) | 800 ms | 800 ms |
| ASR (`small.en`) | 1471 ms | 1681 ms |
| LLM, whole reply (`llama3.2:3b`) | 2896 ms | 11633 ms |
| TTS, whole reply | 392 ms | 1538 ms |
| **Sequential voice-to-voice** | 5534 ms | 15644 ms |
| **Streaming first audio** | **3319 ms** | **3559 ms** |

- **Word error rate:** 16.7% on 9 questions (one speaker). Lower-error model `small` was 11.1%, but about 3× slower.
- **Barge-in reaction time:** about 220 ms from detection to playback stop, plus a 160 ms detection window. Measured on 8 interruptions; one outlier at 662 ms.
- **False triggers:** 0 in 12 clean playbacks (echo test at full speaker volume, plus long answers with no interruption). This sample is small; the true rate could be up to about 25%.

### Model comparison (same 9 questions)

| Model | WER | ASR median | Notes |
|---|---|---|---|
| tiny | 46.3% | 0.3 s | fastest, unusable accuracy |
| base | 25.9% | 0.5-0.6 s | |
| small | 11.1% | 1.5-1.6 s | most accurate, slowest tail |
| base.en | 42.6% | 0.5 s | worse than base here |
| small.en | 14.8% | 1.6-1.75 s | **used** (steady speed) |
| distil-small.en | 24-28% | 1.5 s | only 14% faster, rejected |

### LLM choice

- `llama3.2:1b` was faster, but its memory recall failed in some runs and it invented facts.
- `llama3.2:3b` recalled the name in 5 of 6 answers, with the first-audio time about 5.3 s in live use. It was chosen for recall. Part of its recall is copying its own earlier replies, not only the user's statement.

### Things that did not help

- **Beam size 1:** no speedup over beam size 5.
- **"Very short first sentence" prompt:** no measurable gain.
- **Whisper retries:** disabling them (`temperature=0.0`) cut ASR p95 from 4.8 s to 1.7 s.

## Barge-in rule

The assistant stops when the microphone hears speech that is loud and clearly voice for **5 chunks in a row** (about 160 ms):
- Silero probability ≥ 0.9
- Mic peak ≥ 0.15

The probability alone is not enough. At full speaker volume, the assistant's own voice reached probability 0.9 in many chunks, so the peak gate does most of the work. Soft interruptions like a quiet "wait" are **not** detected.

## Getting Started

Tested on Windows 11 with Python 3.14, CPU only.

```bash
git clone https://github.com/manankundani268/Ultra_Low_Latency_Voice_Pipeline.git
cd Ultra_Low_Latency_Voice_Pipeline
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b

# download the Piper voice (run in the project folder)
python -m piper.download_voices en_US-lessac-medium

python pipeline.py llama3.2:3b
```

Run `python benchmark.py` for the offline benchmark. It needs the 9 test WAVs in `testset/`, which are not included: record your own with `python record_testset.py`, using the questions in `testset/questions.txt`.

**Before timing anything:** plug the laptop in and set Windows power mode to Best performance. On battery, ASR p50 rose from 1.6 s to 2.5 s in our test.

## Limitations

- **One speaker, one laptop, one room.** The benchmark uses 9 questions.
- **Speed:** about 3.3 s on CPU. Streaming ASR and GPU hardware would be far faster; this project measures a CPU baseline and names its bottlenecks.
- **Accuracy:** Whisper mishears short words and names (for example "Ollama" becomes "Olama"). The noise gate also skips some soft real speech.
- **Barge-in:** loud interruptions only. Soft "wait" is missed. Echo was tested on laptop speakers; headphones are the fallback.
- **LLM:** the 3B model sometimes invents facts and sometimes ignores the sentence limit. Memory covers only the last 3 exchanges and is lost on exit.
- **No Docker or frontend yet.**

## What I Learned

- Audio basics: sample rate and Nyquist, int16 vs float32, 1D vs 2D arrays, buffering.
- Real-time design: non-blocking streams, threads with `threading.Event`, a pre-roll buffer so no speech is lost.
- Measurement: p50 vs p95, why one slow run matters, and why a benchmark must control power state and background load.
- Negative results are useful: several "obvious" speedups did not help, and the tests showed it.

## Next Steps

- Dockerize the pipeline.
- Add a simple web frontend.
- Colab GPU timing for the ASR stage, as a separate labeled row.
- Save conversation logs to disk.

## References

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), 2022
- Silero Team, *Silero VAD*, 2021
- Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue*, Kyutai, 2024 (full-duplex, the direction beyond this cascaded design)

## License

Piper TTS is GPL-3.0, so this project should use a GPL-3.0-compatible license. Add a `LICENSE` file with the GPL-3.0 text (for example, from `choosealicense.com`) before publishing.
