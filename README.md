(English | [简体中文](./README_ZH.md))

<p align="center">
  <img src="assets/medasr-banner.svg" width="760" alt="MedASR">
</p>

<p align="center">
  <strong>Low-latency, high-concurrency ASR for long conversations</strong><br>
  <sub>Incremental encoding · Dynamic batching · Long-session recovery · Production-ready API</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/streaming-WebSocket-7B61FF" alt="Streaming WebSocket">
  <img src="https://img.shields.io/badge/vLLM-0.18.0-00A67E" alt="vLLM">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
</p>

---

MedASR is a streaming speech recognition service for long conversations. Built with Qwen3-ASR, FunASR FSMN-VAD, and vLLM, it delivers low-latency, high-concurrency inference. Mandarin and Cantonese are the primary validated languages.

## Features

- **Incremental encoding**: process only new audio while reusing historical features and prefix cache.
- **Concurrent inference**: dynamic cross-session batching and multi-GPU worker scheduling.
- **Long-session optimization**: natural endpoint rollover, pre-roll deduplication, long-silence recovery, and stable incremental text.
- **Service APIs**: HTTP offline transcription, WebSocket streaming, authentication, and secure downloading.

## Quick Start

Set up the environment. [uv](https://docs.astral.sh/uv/) is recommended for environment and dependency management.

```bash
# Set up the environment
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=auto -e '.[gpu,vad]'
sudo apt-get install ffmpeg # Optional: required for offline transcription of various audio formats; not needed for streaming
```

Start the service. Replace `MEDASR_API_KEY` in `.env` with a random secret. Models are downloaded on first startup, and workers may take some time to become ready.

```bash
# Start
cp .env.example .env
bash run.sh --start
```

Other commands:

```bash
bash run.sh --status # Show service status
bash run.sh --stop # Stop the service
bash run.sh --help # Show other commands
```

## Model Downloads

| Component | Hugging Face | ModelScope |
|---|---|---|
| Qwen3-ASR-1.7B | [Download](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | [Download](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) |
| FSMN-Monophone VAD | [Download](https://huggingface.co/funasr/fsmn-vad) | [Download](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) |

The default configuration uses model IDs directly. For offline deployment, download the models in advance and set their absolute paths in `.env`.

Hugging Face:

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
  --local-dir models/Qwen3-ASR-1.7B
huggingface-cli download funasr/fsmn-vad \
  --local-dir models/speech_fsmn_vad-zh-cn-16k-common-pytorch
```

ModelScope:

```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B \
  --local_dir models/Qwen3-ASR-1.7B
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --local_dir models/speech_fsmn_vad-zh-cn-16k-common-pytorch
```

## Usage

Mandarin and Cantonese have been validated; other languages remain unverified. `lang` accepts `zh`, `yue`, or `auto`.

### Offline Transcription

```bash
curl -X POST http://127.0.0.1:18080/v1/asr/offline \
  -H "Authorization: Bearer $MEDASR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.org/audio.wav","lang":"zh"}'
```

### Streaming Transcription

```bash
python examples/ws_client.py sample.wav \
  --url ws://127.0.0.1:18080/ws/asr \
  --api-key "$MEDASR_API_KEY" --lang zh
```

## Configuration

Configuration file: `config/asr_server.yaml`

| Environment variable | Default | Description |
|---|---:|---|
| `MEDASR_API_KEY` | none | Required; business requests are denied when unset |
| `MEDASR_DEFAULT_LANG` | `zh` | Default language: `zh`, `yue`, or `auto` |
| `MEDASR_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Model ID or local path |
| `MEDASR_VAD_MODEL` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | VAD model ID or local path |
| `MEDASR_VAD_ENABLED` | `true` | Enable FSMN-VAD endpoint detection |
| `MEDASR_CARDS` | `auto` | GPUs to use; `auto` starts one worker per visible GPU |
| `MEDASR_GPU_UTIL` | `0.8` | GPU memory fraction per worker |
| `MEDASR_MAX_STREAM_PER_CARD` | `2` | Streaming admission limit per GPU; `0` means unlimited |
| `MEDASR_INCR_ENCODE` | `true` | Enable incremental encoding |
| `MEDASR_MICRO_BATCH` | `true` | Enable dynamic cross-session batching |

## License and Acknowledgements

MedASR is released under the [Apache License 2.0](LICENSE). Dependencies and model weights remain subject to their respective licenses. Thanks to [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [FunASR](https://github.com/modelscope/FunASR), [vLLM](https://github.com/vllm-project/vllm), and their contributors.

```bibtex
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi and Xiong Wang and Zhifang Guo and Yongqi Wang and Pei Zhang and Xinyu Zhang and Zishan Guo and Hongkun Hao and Yu Xi and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}

@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
