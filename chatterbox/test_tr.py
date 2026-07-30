import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device,
    t3_model="v3",
)

text = "Merhaba, bugün Chatterbox Türkçe metinden sese dönüşümünü test ediyoruz."

wav = model.generate(
    text,
    language_id="tr",
)

ta.save(
    "test_turkish.wav",
    wav.cpu(),
    model.sr,
)

print("Ses oluşturuldu: test_turkish.wav")