"""
Sidekick için WebRTC tabanlı karşılıklı konuşma denemesi.

Bu dosya mevcut app_conversation.py içindeki agent, faster-whisper istemcisi
ve Chatterbox istemcisini yeniden kullanır. Normal Gradio uygulamasını
değiştirmez.
"""

import asyncio
import os
import tempfile
import threading
import time
import uuid

import numpy as np
import soundfile as sf
from fastrtc import ReplyOnPause, Stream

from app_conversation import ask_agent, synthesize_speech, transcribe_audio


class AsyncAgentRunner:
    """Sidekick agentını tek ve sürekli açık bir asyncio döngüsünde çalıştırır."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            name="sidekick-agent-loop",
            daemon=True,
        )
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()


agent_runner = AsyncAgentRunner()
turn_lock = threading.Lock()


def save_input_audio(audio):
    """FastRTC ses paketini faster-whisper'a gönderilebilecek WAV'a çevirir."""
    sample_rate, samples = audio
    samples = np.asarray(samples)

    if samples.ndim == 2:
        # FastRTC genellikle (kanal, örnek) biçiminde veri gönderir.
        if samples.shape[0] <= 2:
            samples = samples.mean(axis=0)
        else:
            samples = samples.mean(axis=1)

    samples = np.squeeze(samples)

    input_path = os.path.join(
        tempfile.gettempdir(),
        f"sidekick_webrtc_input_{uuid.uuid4().hex}.wav",
    )
    sf.write(input_path, samples, int(sample_rate))
    return input_path


def yield_audio_file(audio_path, frame_seconds=0.25):
    """
    Chatterbox WAV çıktısını küçük WebRTC paketleri halinde hoparlöre gönderir.

    Chatterbox dosyayı tamamen ürettikten sonra bu fonksiyon başlar; buna rağmen
    ses gönderimi WebRTC üzerinden yapıldığı için mikrofon yükleme bağlantısı
    Gradio Audio.stream gibi kopmaz.
    """
    samples, sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    frame_size = max(int(sample_rate * frame_seconds), 1)

    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]

        if frame.size:
            yield (
                int(sample_rate),
                frame.reshape(1, -1),
            )


def voice_reply(audio):
    """
    Kullanıcı sustuğunda FastRTC tarafından bir kez çağrılır.

    WebRTC ses -> faster-whisper -> Sidekick -> Chatterbox -> WebRTC ses
    """
    with turn_lock:
        total_started = time.perf_counter()
        input_path = save_input_audio(audio)

        try:
            stt_started = time.perf_counter()
            user_text = transcribe_audio(input_path)
            stt_seconds = time.perf_counter() - stt_started
            print(
                f"[WEBRTC PERF] faster-whisper: {stt_seconds:.2f}s | "
                f'text="{user_text}"'
            )
        finally:
            try:
                os.remove(input_path)
            except OSError:
                pass

        if not user_text:
            print("[WEBRTC] Konuşma anlaşılamadı; yeniden dinleniyor.")
            return

        agent_started = time.perf_counter()
        answer, approval_needed, _, _ = agent_runner.run(
            ask_agent(
                user_text,
                conversation_mode=True,
            )
        )
        agent_seconds = time.perf_counter() - agent_started
        print(
            f"[WEBRTC PERF] Agent: {agent_seconds:.2f}s | "
            f'answer="{answer}"'
        )

        if approval_needed:
            answer = (
                "Bu işlem kullanıcı onayı gerektiriyor. "
                "Lütfen normal Sidekick ekranından onayla veya reddet."
            )

        tts_started = time.perf_counter()
        audio_path = synthesize_speech(answer)
        tts_seconds = time.perf_counter() - tts_started
        print(f"[WEBRTC PERF] Chatterbox: {tts_seconds:.2f}s")

        if not audio_path:
            return

        try:
            yield from yield_audio_file(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        print(
            "[WEBRTC PERF] Toplam tur: "
            f"{time.perf_counter() - total_started:.2f}s"
        )


stream = Stream(
    handler=ReplyOnPause(
        voice_reply,
        # Chatterbox hoparlörden konuşurken mikrofon yeni tur başlatmasın.
        can_interrupt=False,
    ),
    modality="audio",
    mode="send-receive",
)


if __name__ == "__main__":
    print("[webrtc] Sidekick konuşma modu: http://127.0.0.1:7861")
    stream.ui.launch(
        server_name="127.0.0.1",
        server_port=7861,
    )