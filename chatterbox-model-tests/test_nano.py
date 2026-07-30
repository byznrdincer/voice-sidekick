import time
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

REFERENCE_AUDIO = "/Users/beyzanurdincer/Downloads/ayhan_12dk_clean.wav"
OUTPUT_FILE = "nano_english_test.wav"

TEXT = (
    "Hello, today we are testing the Chatterbox Nano text to speech model. "
    "This voice was cloned from a reference recording. "
    "The system is running locally on an Apple Silicon computer."
)

print("Nano modeli CPU üzerinde yükleniyor...")
load_start = time.perf_counter()

model = ChatterboxTurboTTS.from_pretrained(
    device="cpu",
    nano=True,
)

load_seconds = time.perf_counter() - load_start
print(f"Model yükleme süresi: {load_seconds:.2f} saniye")

print("Ses üretiliyor...")
generation_start = time.perf_counter()

wav = model.generate(
    TEXT,
    audio_prompt_path=REFERENCE_AUDIO,
)

generation_seconds = time.perf_counter() - generation_start

ta.save(OUTPUT_FILE, wav, model.sr)

audio_seconds = wav.shape[-1] / model.sr
rtf = generation_seconds / audio_seconds if audio_seconds else 0

print("\nTest tamamlandı.")
print(f"Çıktı: {OUTPUT_FILE}")
print(f"Örnekleme hızı: {model.sr} Hz")
print(f"Ses süresi: {audio_seconds:.2f} saniye")
print(f"Üretim süresi: {generation_seconds:.2f} saniye")
print(f"RTF: {rtf:.3f}")

