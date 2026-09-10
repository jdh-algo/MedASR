import argparse
import asyncio
import json
import wave

import websockets


def load_pcm(path):
    try:
        with wave.open(path, "rb") as wav:
            if wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 16000:
                return wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error):
        pass

    import librosa
    import numpy as np

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def transcribe(args):
    pcm = load_pcm(args.audio)

    async with websockets.connect(
        args.url,
        subprotocols=["bearer", args.api_key],
        ping_interval=None,
        proxy=None,
    ) as ws:
        await ws.send(json.dumps({"event": "start", "lang": args.lang, "sample_rate": 16000}))
        frame_bytes = int(16000 * 2 * args.frame_seconds)
        for offset in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[offset:offset + frame_bytes])
            await asyncio.sleep(args.frame_seconds)
        await ws.send(json.dumps({"event": "eof"}))

        async for message in ws:
            event = json.loads(message)
            if event.get("type") == "text":
                print(event.get("delta", ""), end="", flush=True)
            elif event.get("type") == "error":
                raise RuntimeError(event.get("msg", "server error"))
            elif event.get("type") == "done":
                print()
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--url", default="ws://127.0.0.1:18080/ws/asr")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--lang", choices=["zh", "yue", "auto"], default="zh")
    parser.add_argument("--frame-seconds", type=float, default=0.1)
    asyncio.run(transcribe(parser.parse_args()))


if __name__ == "__main__":
    main()
