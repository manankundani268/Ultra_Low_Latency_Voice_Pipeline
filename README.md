# 🎙️ Real-Time Voice-to-Voice Pipeline

> A low-latency voice assistant built from scratch in Python. It listens, understands, and speaks back in real time, with latency measured at every stage.

The goal is not just to chain models together, but to understand and measure where every millisecond goes between the moment a user stops speaking and the moment the assistant's first word is heard.

---

## 📌 Highlights

- **Streaming end to end:** each stage passes partial results onward instead of waiting for the full result
- **Turn detection:** Silero VAD with a tuned silence threshold
- **Barge-in:** the assistant stops talking when the user interrupts
- **Measured, not guessed:** every stage is timed, and every optimization has a before/after number
- **Voice-to-voice latency:** **[X] s → [Y] ms (p50)** on **[hardware]**

---

## 🧩 Architecture

```
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────────┐
│ Mic │ → │ VAD │ → │ ASR │ → │ LLM │ → │ TTS │ → │ Speaker │
└─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────────┘
                                  ▲
        barge-in: stop playback when the user speaks over the reply
```

| Stage | Tool | Role |
|---|---|---|
| Capture / playback | `sounddevice` | Read the mic and play audio in chunks |
| Turn detection | Silero VAD | Detect when the user has finished speaking |
| Speech recognition | `faster-whisper` (int8) | Speech to text |
| Response generation | [LLM runtime] | Streams the reply token by token |
| Speech synthesis | [TTS engine] | Starts speaking after the first complete sentence |

---

## ⏱️ Performance

All stages are instrumented with `time.perf_counter()`. Benchmarks use 50 test utterances and report p50 and p95 voice-to-voice latency.

| Version | p50 (ms) | p95 (ms) | Change |
|---|---|---|---|
| Sequential baseline | [ ] | [ ] | n/a |
| + VAD turn detection | [ ] | [ ] | [ ] |
| + Streaming LLM output | [ ] | [ ] | [ ] |
| + Sentence-level TTS chunking | [ ] | [ ] | [ ] |
| + Quantization | [ ] | [ ] | [ ] |
| + Barge-in handling | [ ] | [ ] | [ ] |

Hardware: **[CPU / GPU / RAM]**

---

## 🗺️ Progress

- [x] **Rung 0:** audio capture, playback, and amplitude checks
- [x] **Rung 1:** slow baseline (record → ASR → LLM → TTS) with per-stage timing
- [x] **Rung 2:** automatic turn-taking with VAD
- [ ] **Rung 3:** streaming and sentence-level chunking
- [ ] **Rung 4:** barge-in and echo handling
- [ ] **Rung 5:** benchmarks over 50 utterances, optional GPU optimization

---

## 🛠️ Tech Stack

Python · sounddevice · NumPy · soundfile · Silero VAD · faster-whisper · [LLM runtime] · [TTS engine]

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/realtime-voice-pipeline.git
cd realtime-voice-pipeline
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
python pipeline.py
```

**Requirements:** a working microphone and speakers, Python 3.10+, [GPU requirements, if any].

> **Troubleshooting:** if the recording peak is near `0.0`, the mic is muted or blocked by the OS. Check the default input with `sd.query_devices()`.

---

## 📚 What I Learned

- Audio fundamentals: sample rate, Nyquist, int16 vs float32, buffering
- Real-time streaming design and non-blocking audio callbacks
- Latency measurement methodology, and why p95 matters more than the average for a natural-feeling conversation

---

## ⚠️ Limitations and Future Work

- Latency depends heavily on hardware and model choice. Results here were measured on **[hardware]**.
- Planned: a narrow vision feature (send one camera frame only when the question needs it) and a comparison with end-to-end speech models such as Moshi.

---

## 📖 References

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), 2022
- Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue*, Kyutai, 2024
- Fang et al., *LLaMA-Omni: Seamless Speech Interaction with Large Language Models*, 2024

---
