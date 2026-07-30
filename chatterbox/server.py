from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import uuid

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


app = FastAPI()

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"[chatterbox] using device: {device}")

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device,
    t3_model="v3",
)


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def tts(request: TTSRequest):
    output_path = (
        tempfile.gettempdir()
        + f"/chatterbox_{uuid.uuid4().hex}.wav"
    )

    wav = model.generate(
        request.text,
        language_id="tr",
        audio_prompt_path="/Users/beyzanurdincer/Downloads/sidekick_reference_full.wav",
    )

    ta.save(
        output_path,
        wav.cpu(),
        model.sr,
    )

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="speech.wav",
    )