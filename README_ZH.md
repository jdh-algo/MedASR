([English](./README.md) | 简体中文)

<p align="center">
  <img src="assets/medasr-banner.svg" width="760" alt="MedASR">
</p>

<p align="center">
  <strong>面向长时对话的低延迟、高并发 ASR 服务</strong><br>
  <sub>Incremental encoding · Dynamic batching · Long-session recovery · Production-ready API</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/streaming-WebSocket-7B61FF" alt="Streaming WebSocket">
  <img src="https://img.shields.io/badge/vLLM-0.18.0-00A67E" alt="vLLM">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
</p>

---

MedASR 是一个面向长时对话场景的流式语音识别服务，基于 Qwen3-ASR 和 FunASR FSMN-VAD 构建，并使用 vLLM 提供低延迟、高并发推理。已在普通话和粤语场景完成验证。

## 特性

- **增量编码**：仅处理新增音频，复用历史特征与前缀缓存，降低长会话重复计算。
- **高并发推理**：支持跨会话动态微批和多 GPU worker 调度。
- **长会话优化**：支持自然端点换段、pre-roll 去重、长静默恢复和稳定增量文本。
- **服务化接口**：提供 HTTP 离线识别、WebSocket 流式识别、鉴权和安全下载。

## 快速开始

配置环境：推荐使用 [uv](https://docs.astral.sh/uv/) 管理环境和依赖。

```bash
# 配置环境
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=auto -e '.[gpu,vad]'
sudo apt-get install ffmpeg # 【可选】离线识别需要使用ffmpeg处理各种音频格式，流式不需要
```

启动服务：将 .env 中的 MEDASR_API_KEY 换成随机密钥。首次启动会自动下载模型。模型较大，worker 就绪需要一定时间。
```bash
# 启动
cp .env.example .env
bash run.sh --start
```

其他命令
```bash
bash run.sh --status # 查看服务状态
bash run.sh --stop # 停止服务
bash run.sh --help # 查看其他命令
```



## 模型下载

| 组件 | Hugging Face | ModelScope |
|---|---|---|
| Qwen3-ASR-1.7B | [下载](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | [下载](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) |
| FSMN-Monophone VAD | [下载](https://huggingface.co/funasr/fsmn-vad) | [下载](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) |

默认配置直接使用模型 ID。离线环境可以预先下载，并在 `.env` 中配置本地绝对路径。

Hugging Face

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
  --local-dir models/Qwen3-ASR-1.7B
huggingface-cli download funasr/fsmn-vad \
  --local-dir models/speech_fsmn_vad-zh-cn-16k-common-pytorch
```

ModelScope

```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B \
  --local_dir models/Qwen3-ASR-1.7B
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --local_dir models/speech_fsmn_vad-zh-cn-16k-common-pytorch
```

## 使用方式

已验证普通话和粤语，其他待验证：`lang` 支持 `zh`、`yue` 和 `auto`。

### 离线识别

```bash
curl -X POST http://127.0.0.1:18080/v1/asr/offline \
  -H "Authorization: Bearer $MEDASR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.org/audio.wav","lang":"zh"}'
```

### 流式识别

```bash
python examples/ws_client.py sample.wav \
  --url ws://127.0.0.1:18080/ws/asr \
  --api-key "$MEDASR_API_KEY" --lang zh
```

## 配置

配置文件：`config/asr_server.yaml`


| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MEDASR_API_KEY` | 无 | 必填；未设置时拒绝请求 |
| `MEDASR_DEFAULT_LANG` | `zh` | 默认语言：`zh`、`yue` 或 `auto` |
| `MEDASR_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | 模型 ID 或本地路径 |
| `MEDASR_VAD_MODEL` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | VAD 模型 ID 或本地路径 |
| `MEDASR_VAD_ENABLED` | `true` | 是否启用 FSMN-VAD 端点检测 |
| `MEDASR_CARDS` | `auto` | 使用的 GPU；`auto` 表示每张可见卡启动一个 worker |
| `MEDASR_GPU_UTIL` | `0.8` | 每个 worker 的显存比例 |
| `MEDASR_MAX_STREAM_PER_CARD` | `2` | 每卡流式并发上限；`0` 表示不限 |
| `MEDASR_INCR_ENCODE` | `true` | 是否启用增量编码 |
| `MEDASR_MICRO_BATCH` | `true` | 是否启用跨会话动态微批 |

## 许可与致谢

MedASR 使用 [Apache License 2.0](LICENSE)。依赖和模型权重分别遵循各自许可证。感谢 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[FunASR](https://github.com/modelscope/FunASR) 和 [vLLM](https://github.com/vllm-project/vllm) 项目及其贡献者。

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
