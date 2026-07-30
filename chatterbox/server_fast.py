import os
import tempfile
import threading
import time
import uuid

import torch
import torchaudio as ta
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


app = FastAPI()

REFERENCE_AUDIO = (
    "/Users/beyzanurdincer/Downloads/sidekick_reference_full.wav"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[chatterbox] using device: {device}")


# Model yalnızca bir kez yüklenir.
model = ChatterboxMultilingualTTS.from_pretrained(
    device=device,
    t3_model="v3",
)


# Referans sesin speaker embedding ve diğer condition değerleri yalnızca
# başlangıçta hazırlanır. /tts çağrılarında tekrar işlenmez.
print("[chatterbox] preparing reference voice once...")
prepare_started = time.perf_counter()
model.prepare_conditionals(
    REFERENCE_AUDIO,
    exaggeration=0.50,
)

if device == "mps":
    torch.mps.synchronize()

print(
    "[chatterbox] reference ready in "
    f"{time.perf_counter() - prepare_started:.2f}s"
)


# Aynı model üzerinde iki üretimin aynı anda başlamasını engeller.
generation_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


@app.post("/tts")
def tts(request: TTSRequest):
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"chatterbox_{uuid.uuid4().hex}.wav",
    )

    started = time.perf_counter()

    with generation_lock, torch.inference_mode():
        wav = model.generate(
            request.text,
            language_id="tr",
            # audio_prompt_path bilerek verilmez:
            # başlangıçta hazırlanan conditionals yeniden kullanılır.
            exaggeration=0.50,
            cfg_weight=0.30,
            temperature=0.8,
            repetition_penalty=1.0,
        )

        if device == "mps":
            torch.mps.synchronize()

    ta.save(
        output_path,
        wav.cpu(),
        model.sr,
    )

    elapsed = time.perf_counter() - started
    audio_seconds = wav.shape[-1] / float(model.sr)
    realtime_factor = elapsed / max(audio_seconds, 0.001)

    print(
        f"[chatterbox] text={len(request.text)} chars | "
        f"audio={audio_seconds:.2f}s | "
        f"generation={elapsed:.2f}s | "
        f"RTF={realtime_factor:.2f}"
    )

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(remove_file, output_path),
    )