import asyncio
import os
import re
import threading
import time
import uuid
import tempfile

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from fastrtc import ReplyOnPause, Stream
from langgraph.types import Command

from sidekick.agent import build_agent
from speech.tts_text_processor import TTSTextProcessor

load_dotenv()


agent = None
thread_id = str(uuid.uuid4())
pending_interrupt = False

tts_text_processor = TTSTextProcessor()


# ============================================================
# AGENT
# ============================================================

async def initialize():
    global agent

    if agent is None:
        agent = await build_agent()


def make_fallback_todos(message: str):
    """Çok adımlı kullanıcı görevinden kısa bir Todo planı çıkarır."""

    text = message.strip()

    # Büyük/küçük harf fark etmeden görev geçişlerini ayır.
    parts = re.split(
        r"(?i)\b(?:önce|sonra|ardından|daha sonra|son olarak|ve son olarak)\b",
        text,
    )

    cleaned = []

    for part in parts:
        # Nokta ile ayrılmış ek görevleri de yakala.
        subparts = re.split(r"[.;]\s*", part)

        for item in subparts:
            item = item.strip(" ,.-:\n\t")

            if len(item) < 5:
                continue

            # Gereksiz giriş ifadelerini temizle.
            item = re.sub(
                r"(?i)^(bu görev için|bu projelerden|sonuç olarak)\s*",
                "",
                item,
            ).strip()

            if item:
                cleaned.append(item)

    # Tekrarları kaldır.
    unique = []
    for item in cleaned:
        if item not in unique:
            unique.append(item)

    if len(unique) < 2:
        return []

    return [
        {
            "content": item[:85],
            "status": "completed",
        }
        for item in unique[:8]
    ]



def build_runtime_graph(messages):
    """
    LangChain message geçmişinden gerçek execution katmanlarını çıkarır.

    Her AIMessage bir model turudur.
    Aynı AIMessage içindeki birden fazla tool_call gerçekten aynı
    model turundan çıktığı için aynı graph katmanında dallandırılır.
    """
    layers = [
        {
            "type": "user",
            "items": ["Kullanıcı"],
        }
    ]

    model_turn = 0

    for msg in messages:
        msg_type = getattr(msg, "type", "")

        # Sadece AI/model mesajları execution kararını temsil ediyor.
        if msg_type not in ("ai", "assistant"):
            continue

        model_turn += 1
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            # Önce o kararın verildiği model turu
            layers.append(
                {
                    "type": "model",
                    "items": [f"Model {model_turn}"],
                }
            )

            # Aynı AIMessage içindeki tool'lar aynı katmanda.
            tool_names = []

            for call in tool_calls:
                if isinstance(call, dict):
                    name = call.get("name", "Tool")
                else:
                    name = getattr(call, "name", "Tool")

                tool_names.append(name)

            layers.append(
                {
                    "type": "tools",
                    "items": tool_names,
                }
            )

        else:
            # Tool çağrısı olmayan son AI mesajı final cevaptır.
            layers.append(
                {
                    "type": "final",
                    "items": ["Final Answer"],
                }
            )

    # Herhangi bir nedenle final AIMessage bulunamadıysa güvenli fallback.
    if not layers or layers[-1]["type"] != "final":
        layers.append(
            {
                "type": "final",
                "items": ["Final Answer"],
            }
        )

    return layers


def runtime_graph_html(layers):
    """Gerçek model turu -> tool çağrısı yapısını SVG graph olarak çizer."""

    width = 1200
    center_x = width / 2

    node_w = 230
    node_h = 62
    layer_gap = 145
    top = 70

    height = top + max(len(layers) - 1, 1) * layer_gap + 110

    # -------------------------------------------------------
    # Her katmandaki node koordinatlarını hesapla
    # -------------------------------------------------------
    positions = []

    for index, layer in enumerate(layers):
        y = top + index * layer_gap
        items = layer["items"]

        if len(items) == 1:
            xs = [center_x]

        else:
            count = len(items)

            # Çok fazla tool varsa taşmayı azalt.
            usable_width = min(900, 250 * (count - 1))
            start_x = center_x - usable_width / 2
            spacing = usable_width / (count - 1)

            xs = [
                start_x + i * spacing
                for i in range(count)
            ]

        positions.append(
            [
                {
                    "label": label,
                    "x": x,
                    "y": y,
                    "type": layer["type"],
                }
                for label, x in zip(items, xs)
            ]
        )

    svg = [
        f'''
        <svg
            viewBox="0 0 {width} {height}"
            width="100%"
            xmlns="http://www.w3.org/2000/svg"
        >
            <defs>
                <marker
                    id="runtime-arrow"
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto"
                >
                    <path
                        d="M 0 0 L 10 5 L 0 10 z"
                        fill="#94a3b8"
                    />
                </marker>
            </defs>
        '''
    ]

    # -------------------------------------------------------
    # GERÇEK katmanlar arasındaki edge'leri çiz
    # -------------------------------------------------------
    for layer_index in range(len(positions) - 1):

        current_layer = positions[layer_index]
        next_layer = positions[layer_index + 1]

        for source in current_layer:
            for target in next_layer:

                x1 = source["x"]
                y1 = source["y"] + node_h / 2

                x2 = target["x"]
                y2 = target["y"] - node_h / 2

                if abs(x1 - x2) < 5:
                    path = f"M {x1} {y1} L {x2} {y2}"

                else:
                    middle_y = (y1 + y2) / 2

                    path = (
                        f"M {x1} {y1} "
                        f"C {x1} {middle_y}, "
                        f"{x2} {middle_y}, "
                        f"{x2} {y2}"
                    )

                svg.append(
                    f'''
                    <path
                        d="{path}"
                        fill="none"
                        stroke="#94a3b8"
                        stroke-width="2.3"
                        marker-end="url(#runtime-arrow)"
                    />
                    '''
                )

    # -------------------------------------------------------
    # Node'ları çiz
    # -------------------------------------------------------
    tool_counter = 0

    for layer in positions:
        for node in layer:

            label = node["label"]
            node_type = node["type"]
            x = node["x"]
            y = node["y"]

            if node_type == "user":
                fill = "#fff7ed"
                stroke = "#f97316"

            elif node_type == "model":
                fill = "#f8fafc"
                stroke = "#94a3b8"

            elif node_type == "final":
                fill = "#f0fdf4"
                stroke = "#22c55e"

            else:
                fill = "#eff6ff"
                stroke = "#60a5fa"

                tool_counter += 1
                label = f"{tool_counter}. {label}"

            # Uzun isimler için biraz daha geniş tool kutusu
            current_w = node_w

            if len(label) > 23:
                current_w = 270

            svg.append(
                f'''
                <rect
                    x="{x - current_w / 2}"
                    y="{y - node_h / 2}"
                    width="{current_w}"
                    height="{node_h}"
                    rx="16"
                    fill="{fill}"
                    stroke="{stroke}"
                    stroke-width="2"
                />
                '''
            )

            svg.append(
                f'''
                <text
                    x="{x}"
                    y="{y + 5}"
                    text-anchor="middle"
                    font-family="Inter, Arial, sans-serif"
                    font-size="15"
                    font-weight="700"
                    fill="#1e293b"
                >
                    {label}
                </text>
                '''
            )

    svg.append("</svg>")

    return f"""
    <div style="
        background:white;
        border:1px solid #e2e8f0;
        border-radius:22px;
        padding:26px;
        overflow:auto;
    ">

        <div style="
            text-align:center;
            font-size:25px;
            font-weight:800;
            color:#1e293b;
        ">
            Son Görevin Gerçek Çalışma Grafiği
        </div>

        <div style="
            text-align:center;
            color:#64748b;
            margin:6px 0 18px 0;
        ">
            Her Model düğümü gerçek bir AIMessage turunu,
            dallar ise o turda yapılan gerçek tool çağrılarını gösterir.
        </div>

        {''.join(svg)}

    </div>
    """

async def ask_agent(message: str, conversation_mode: bool = False):

    global agent, pending_interrupt

    await initialize()

    agent_message = message

    if conversation_mode:
        agent_message = (
            f"{message}\n\n"
            "[SİSTEM NOTU: Bu istek sesli ve karşılıklı konuşma modundan geldi. "
            "Yanıtını doğal Türkçeyle, en fazla iki kısa cümle halinde ver. "
            "Gereksiz giriş, tekrar, uzun açıklama ve madde listesi kullanma. "
            "Kullanıcı özellikle ayrıntı isterse yalnızca gerekli ayrıntıyı ver.]"
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": agent_message
                }
            ]
        },
        config=config,
    )

    # Human-in-the-Loop interrupt
    if result.get("__interrupt__"):
        pending_interrupt = True

        interrupt = result["__interrupt__"][0]
        value = interrupt.value

        action = value["action_requests"][0]

        name = action["name"]
        args = action["args"]

        text = (
            "Bu işlem senin onayını gerektiriyor.\n\n"
            f"**İşlem:** `{name}`\n\n"
            f"**Parametreler:**\n`{args}`\n\n"
            "İşlemi onaylayabilir veya reddedebilirsin."
        )

        return text, True, [], runtime_graph_html(["Kullanıcı", "Model", "Human Approval"])

    pending_interrupt = False

    todos = result.get("todos", [])

    if not todos:
        todos = make_fallback_todos(message)

    steps = build_runtime_graph(result["messages"])
    graph_html = runtime_graph_html(steps)

    return result["messages"][-1].content, False, todos, graph_html


# ============================================================
# SPEECH
# ============================================================

def transcribe_audio(audio_path: str) -> str:
    """
    Mikrofon kaydını ayrı çalışan local OpenAI Whisper
    servisine gönderir ve dönen Türkçe metni alır.
    """

    if not audio_path:
        return ""

    with open(audio_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8002/transcribe",
            files={
                "file": (
                    os.path.basename(audio_path),
                    f,
                    "application/octet-stream",
                )
            },
            timeout=300,
        )

    response.raise_for_status()

    data = response.json()
    return data.get("text", "").strip()


def _merge_wav_files(wav_paths, output_path):
    """
    Chatterbox float WAV parçalarını tek WAV dosyasında birleştirir.
    Chatterbox çıktıları IEEE float WAV olduğu için standart wave modülü
    yerine soundfile kullanılır.
    """
    import numpy as np
    import soundfile as sf

    if not wav_paths:
        raise RuntimeError("Birleştirilecek TTS çıktısı bulunamadı.")

    audios = []
    sample_rate = None

    for path in wav_paths:
        audio, sr = sf.read(path)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                "Chatterbox WAV parçalarının sample rate değerleri eşleşmiyor."
            )

        audios.append(audio)

    combined = np.concatenate(audios)
    sf.write(output_path, combined, sample_rate)


def synthesize_speech(text: str):
    """
    Sidekick cevabını local Chatterbox Multilingual V3 servisiyle sese çevirir.

    Pipeline:
    Agent cevabı
        -> TTS için düz/konuşulabilir metin hazırlama
        -> Türkçe sayı/saat/yüzde normalizasyonu
        -> doğal TTS chunk'ları
        -> her chunk için Chatterbox /tts
        -> WAV parçalarını birleştirme
        -> tek ses dosyası

    Chatterbox server:
        http://127.0.0.1:8001/tts
    """

    if not text or not text.strip():
        return None

    speech_text = tts_text_processor.prepare(text)
    chunks = tts_text_processor.split(speech_text, max_chars=400)

    output_dir = tempfile.gettempdir()
    final_path = os.path.join(
        output_dir,
        f"sidekick_chatterbox_{uuid.uuid4().hex}.wav",
    )

    chunk_paths = []

    try:
        for index, chunk in enumerate(chunks):
            response = requests.post(
                "http://127.0.0.1:8001/tts",
                json={"text": chunk},
                timeout=300,
            )
            response.raise_for_status()

            chunk_path = os.path.join(
                output_dir,
                f"sidekick_chatterbox_chunk_{uuid.uuid4().hex}_{index}.wav",
            )

            with open(chunk_path, "wb") as f:
                f.write(response.content)

            chunk_paths.append(chunk_path)

        if not chunk_paths:
            raise RuntimeError("Chatterbox hiçbir ses çıktısı üretmedi.")

        if len(chunk_paths) == 1:
            os.replace(chunk_paths[0], final_path)
            chunk_paths = []
        else:
            _merge_wav_files(chunk_paths, final_path)

        return final_path

    finally:
        for chunk_path in chunk_paths:
            try:
                os.remove(chunk_path)
            except OSError:
                pass


# ============================================================
# CHAT
# ============================================================

async def handle_message(
    message,
    history,
    voice_enabled,
    conversation_mode=False,
):
    history = history or []

    if not message or not message.strip():
        return (
            history,
            "",
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            "### Yapılacaklar\nHenüz plan oluşturulmadı.",
            "<div style='padding:20px;text-align:center;color:#64748b;'>Henüz bir görev çalıştırılmadı.</div>",
        )

    agent_started = time.perf_counter()
    answer, approval_needed, todos, runtime_graph = await ask_agent(
        message,
        conversation_mode=conversation_mode,
    )
    agent_seconds = time.perf_counter() - agent_started
    print(f"[PERF] Agent: {agent_seconds:.2f} saniye")

    history = history + [
        {
            "role": "user",
            "content": message
        },
        {
            "role": "assistant",
            "content": answer
        },
    ]

    # Onay bekleyen mesajı TTS ile okumaya gerek yok
    audio_out = None

    if voice_enabled and not approval_needed:
        tts_started = time.perf_counter()
        audio_out = synthesize_speech(answer)
        tts_seconds = time.perf_counter() - tts_started
        print(f"[PERF] Chatterbox TTS: {tts_seconds:.2f} saniye")

    if todos:
        todo_lines = ["### Yapılacaklar", ""]
        for todo in todos:
            status = todo.get("status", "")
            content = todo.get("content", "")
            icon = "✓" if status == "completed" else "○"
            todo_lines.append(f"- {icon} {content}")
        todo_text = "\n".join(todo_lines)
    else:
        todo_text = "### Yapılacaklar\nHenüz plan oluşturulmadı."

    return (
        history,
        "",
        audio_out,
        gr.update(visible=approval_needed),
        gr.update(visible=approval_needed),
        todo_text,
        runtime_graph,
    )


async def handle_text(message, history, voice_enabled):
    return await handle_message(
        message,
        history,
        voice_enabled
    )


async def handle_audio(audio_path, history, voice_enabled):
    if not audio_path:
        return (
            history,
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            "### Yapılacaklar\nHenüz plan oluşturulmadı.",
            "<div style='padding:20px;text-align:center;color:#64748b;'>Henüz bir görev çalıştırılmadı.</div>",
        )

    user_text = transcribe_audio(audio_path)

    result = await handle_message(
        user_text,
        history,
        voice_enabled
    )

    history, _, audio_out, approve_update, reject_update, todo_text, runtime_graph = result

    return (
        history,
        None,
        audio_out,
        approve_update,
        reject_update,
        todo_text,
        runtime_graph,
    )


def new_conversation_state():
    """Sürekli mikrofon akışında tek bir konuşma turunun durumunu tutar."""
    return {
        "sample_rate": None,
        "chunks": [],
        "speech_started": False,
        "silence_seconds": 0.0,
        "blocked_until": 0.0,
    }


def _empty_conversation_update(history, state, status):
    """Henüz tamamlanmış bir cümle yokken arayüzü değiştirmeden devam eder."""
    return (
        history,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        state,
        status,
    )


def _audio_duration(audio_path):
    if not audio_path:
        return 0.0

    try:
        return float(sf.info(audio_path).duration)
    except Exception:
        return 0.0


async def handle_conversation_stream(
    audio_chunk,
    history,
    voice_enabled,
    state,
):
    """
    Açık kalan mikrofondan gelen parçaları biriktirir.

    Konuşma başladıktan sonra yaklaşık 1,2 saniye sessizlik oluştuğunda:
    mikrofon -> faster-whisper -> agent -> Chatterbox akışını otomatik çalıştırır.
    """
    history = history or []
    state = state or new_conversation_state()

    # Sidekick'in hoparlörden çıkan kendi sesini yeniden kullanıcı sesi
    # olarak algılamasını önle.
    if time.monotonic() < state.get("blocked_until", 0.0):
        return _empty_conversation_update(
            history,
            state,
            "🔊 Sidekick konuşuyor…",
        )

    if audio_chunk is None:
        return _empty_conversation_update(
            history,
            state,
            "🟢 Dinliyorum…",
        )

    sample_rate, audio = audio_chunk
    audio = np.asarray(audio)

    if audio.size == 0:
        return _empty_conversation_update(
            history,
            state,
            "🟢 Dinliyorum…",
        )

    # Stereo kayıt gelirse mono hâle getir.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Gradio bazı ortamlarda int16, bazılarında float ses döndürebilir.
    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(float(np.iinfo(audio.dtype).max), 1.0)
        audio_float = audio.astype(np.float32) / max_value
    else:
        audio_float = audio.astype(np.float32)

    audio_float = np.clip(audio_float, -1.0, 1.0)
    chunk_seconds = len(audio_float) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(audio_float)) + 1e-12))

    # Ortam gürültüsüne göre gerekirse 0.015 değeri değiştirilebilir.
    speech_detected = rms >= 0.015

    if not state["speech_started"]:
        if not speech_detected:
            return _empty_conversation_update(
                history,
                state,
                "🟢 Dinliyorum…",
            )

        state["speech_started"] = True
        state["sample_rate"] = sample_rate
        state["chunks"] = [audio_float]
        state["silence_seconds"] = 0.0

        return _empty_conversation_update(
            history,
            state,
            "🎙️ Seni dinliyorum…",
        )

    # Konuşma başladıktan sonra gelen parçaları sessizlik dahil biriktir.
    state["chunks"].append(audio_float)

    if speech_detected:
        state["silence_seconds"] = 0.0
        status = "🎙️ Seni dinliyorum…"
    else:
        state["silence_seconds"] += chunk_seconds
        status = "⏳ Cümlenin bitmesini bekliyorum…"

    if state["silence_seconds"] < 1.2:
        return _empty_conversation_update(history, state, status)

    full_audio = np.concatenate(state["chunks"])
    captured_rate = state["sample_rate"] or sample_rate

    # Yeni tur için kayıt durumunu, uzun süren agent/TTS çağrısından önce sıfırla.
    state["sample_rate"] = None
    state["chunks"] = []
    state["speech_started"] = False
    state["silence_seconds"] = 0.0

    # Çok kısa sesleri yanlış tetikleme olarak kabul et.
    if len(full_audio) / float(captured_rate) < 0.5:
        return _empty_conversation_update(
            history,
            state,
            "🟢 Dinliyorum…",
        )

    input_path = os.path.join(
        tempfile.gettempdir(),
        f"sidekick_conversation_{uuid.uuid4().hex}.wav",
    )
    sf.write(input_path, full_audio, captured_rate)

    stt_started = time.perf_counter()

    try:
        user_text = transcribe_audio(input_path)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass

    stt_seconds = time.perf_counter() - stt_started
    print(f"[PERF] faster-whisper STT: {stt_seconds:.2f} saniye")

    if not user_text:
        return _empty_conversation_update(
            history,
            state,
            "Konuşma anlaşılamadı; dinlemeye devam ediyorum.",
        )

    turn_started = time.perf_counter()
    result = await handle_message(
        user_text,
        history,
        voice_enabled,
        conversation_mode=True,
    )
    turn_seconds = time.perf_counter() - turn_started
    print(f"[PERF] Agent + TTS toplam: {turn_seconds:.2f} saniye")
    (
        updated_history,
        _,
        audio_out,
        approve_update,
        reject_update,
        todo_text,
        runtime_graph,
    ) = result

    # TTS oynarken mikrofon akışı açık kalsa bile model kendi cevabını dinlemesin.
    state["blocked_until"] = (
        time.monotonic()
        + _audio_duration(audio_out)
        + 1.0
    )

    return (
        updated_history,
        audio_out,
        approve_update,
        reject_update,
        todo_text,
        runtime_graph,
        state,
        f'✅ Algılanan konuşma: "{user_text}"',
    )


# ============================================================
# HUMAN-IN-THE-LOOP
# ============================================================

async def approve_action(history, voice_enabled):
    global agent, pending_interrupt

    await initialize()

    if not pending_interrupt:
        return (
            history,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await agent.ainvoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "approve"
                    }
                ]
            }
        ),
        config=config,
    )

    pending_interrupt = False

    answer = result["messages"][-1].content

    history = (history or []) + [
        {
            "role": "assistant",
            "content": answer
        }
    ]

    audio_out = (
        synthesize_speech(answer)
        if voice_enabled
        else None
    )

    return (
        history,
        audio_out,
        gr.update(visible=False),
        gr.update(visible=False),
    )


async def reject_action(history, voice_enabled):
    global agent, pending_interrupt

    await initialize()

    if not pending_interrupt:
        return (
            history,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await agent.ainvoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": "Kullanıcı bu işlemi reddetti."
                    }
                ]
            }
        ),
        config=config,
    )

    pending_interrupt = False

    answer = result["messages"][-1].content

    history = (history or []) + [
        {
            "role": "assistant",
            "content": answer
        }
    ]

    audio_out = (
        synthesize_speech(answer)
        if voice_enabled
        else None
    )

    return (
        history,
        audio_out,
        gr.update(visible=False),
        gr.update(visible=False),
    )



# ============================================================
# AGENT GRAPH
# ============================================================

async def get_agent_graph_html():
    """Gerçek LangGraph node ve edge yapısını dallanmalı SVG olarak göster."""
    await initialize()

    graph = agent.get_graph()
    nodes = list(graph.nodes.keys())
    edges = list(graph.edges)

    def clean(name):
        name = str(name)
        name = name.replace("\\2e", ".")
        name = name.replace("\\5b", "[")
        name = name.replace("\\5d", "]")

        mapping = {
            "__start__": "START",
            "__end__": "END",
            "model": "Model",
            "tools": "Tools",
            "TodoListMiddleware.after_model": "Todo List",
            "PIIMiddleware[email].before_model": "PII Before",
            "PIIMiddleware[email].after_model": "PII After",
            "ModelCallLimitMiddleware.before_model": "Call Limit Before",
            "ModelCallLimitMiddleware.after_model": "Call Limit After",
            "HumanInTheLoopMiddleware.after_model": "Human Approval",
        }

        return mapping.get(name, name)

    # Gerçek node'ları insan-okunabilir isimleriyle eşleştir.
    by_label = {clean(node): node for node in nodes}

    # Dallanmayı gösterecek katmanlı koordinatlar.
    layout = {
        "START": (500, 60),

        "PII Before": (500, 160),
        "Call Limit Before": (500, 260),

        "Model": (500, 370),
        "Human Approval": (500, 480),

        "Call Limit After": (500, 590),
        "PII After": (500, 700),

        "Todo List": (500, 810),

        "Tools": (210, 950),
        "END": (790, 950),
    }

    # Versiyon farkından dolayı bilinmeyen node olursa kaybetme.
    extra_y = 1080
    for label in by_label:
        if label not in layout:
            layout[label] = (500, extra_y)
            extra_y += 100

    width = 1000
    height = max(y for _, y in layout.values()) + 100

    svg = [
        f'<svg viewBox="0 0 {width} {height}" '
        f'width="100%" xmlns="http://www.w3.org/2000/svg">'
    ]

    svg.append("""
    <defs>
      <marker id="arrow" viewBox="0 0 10 10"
              refX="9" refY="5"
              markerWidth="7" markerHeight="7"
              orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>
      </marker>
    </defs>
    """)

    # ------------------------------------------------------
    # GERÇEK EDGE'LER
    # ------------------------------------------------------
    for edge in edges:
        source_label = clean(edge.source)
        target_label = clean(edge.target)

        if source_label not in layout or target_label not in layout:
            continue

        x1, y1 = layout[source_label]
        x2, y2 = layout[target_label]

        sy = y1 + 31
        ty = y2 - 31

        # Aynı kolondaysa düz.
        if abs(x1 - x2) < 30:
            path = f"M {x1} {sy} L {x2} {ty}"
        else:
            # Dallanma / loop için eğri.
            middle = (sy + ty) / 2
            path = (
                f"M {x1} {sy} "
                f"C {x1} {middle}, "
                f"{x2} {middle}, "
                f"{x2} {ty}"
            )

        conditional = bool(getattr(edge, "conditional", False))
        dash = 'stroke-dasharray="8 7"' if conditional else ""

        svg.append(
            f'<path d="{path}" fill="none" '
            f'stroke="#94a3b8" stroke-width="2.3" '
            f'{dash} marker-end="url(#arrow)" />'
        )

    # ------------------------------------------------------
    # NODE'LAR
    # ------------------------------------------------------
    for label in by_label:
        x, y = layout[label]

        fill = "#fafafa"
        stroke = "#cbd5e1"

        if label == "START":
            fill, stroke = "#f1f5f9", "#94a3b8"

        elif label == "END":
            fill, stroke = "#f1f5f9", "#94a3b8"

        elif label == "Model":
            fill, stroke = "#fff7ed", "#f97316"

        elif label == "Tools":
            fill, stroke = "#eff6ff", "#60a5fa"

        elif label == "Todo List":
            fill, stroke = "#f0fdf4", "#4ade80"

        elif label == "Human Approval":
            fill, stroke = "#fef2f2", "#f87171"

        box_w = 220
        box_h = 62

        svg.append(
            f'<rect x="{x-box_w/2}" y="{y-box_h/2}" '
            f'width="{box_w}" height="{box_h}" rx="15" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

        svg.append(
            f'<text x="{x}" y="{y+5}" text-anchor="middle" '
            f'font-family="Inter,Arial,sans-serif" '
            f'font-size="15" font-weight="650" fill="#1e293b">'
            f'{label}</text>'
        )

    svg.append("</svg>")

    return (
        '<div style="background:white;'
        'border:1px solid #e2e8f0;'
        'border-radius:22px;'
        'padding:24px;overflow:auto;">'

        '<div style="text-align:center;'
        'font-size:24px;font-weight:800;color:#1e293b;">'
        'Sidekick LangGraph</div>'

        '<div style="text-align:center;color:#64748b;'
        'margin:6px 0 20px 0;">'
        'Bağlantılar doğrudan agent.get_graph().edges üzerinden çiziliyor.'
        '</div>'

        + "".join(svg) +
        '</div>'
    )

# ============================================================
# UI
# ============================================================

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif"
    ],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 8px 30px rgba(0,0,0,0.06)",
    block_radius="20px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    input_radius="14px",
)


CSS = """
/* Uygulamayı her zaman açık (light) temada tut; sistem/tarayıcı karanlık
   modda olsa bile arka planlar beyaz kaldığı için yazılar kayboluyordu. */
:root, .gradio-container {
    color-scheme: light !important;
}

.gradio-container {
    max-width: 1450px !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding-bottom: 30px !important;
    background: #f8fafc !important;
}

body, .gradio-container, .gradio-container * {
    color: #1e293b;
}

#header {
    text-align: center;
    padding: 24px 0 20px 0;
}

#header h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f97316, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}

#header p {
    font-size: 1.05rem;
    color: #475569 !important;
}

/* ===========================
   CHAT + SIDEBAR ANA ALANI
   =========================== */

#main-workspace {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 22px !important;
    width: 100% !important;
}

#chat-card {
    flex: 2 1 0 !important;
    min-width: 0 !important;
    height: 650px !important;
    background: white !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06) !important;
    overflow: hidden !important;
}

#chatbot {
    height: 650px !important;
    min-height: 650px !important;
    border: none !important;
    border-radius: 24px !important;
    background: white !important;
    color: #1e293b !important;
}

#chatbot * {
    color: #1e293b !important;
}

/* Kullanıcı mesajı */
#chatbot .message.user {
    background: #fff7ed !important;
    border: 1px solid #fed7aa !important;
    border-radius: 16px !important;
}

/* Sidekick mesajı */
#chatbot .message.bot {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
}

/* ===========================
   SAĞ SIDEBAR (Yapılacaklar / Çalışma Akışı / Agent Graph)
   =========================== */

#sidebar {
    flex: 1 1 0 !important;
    min-width: 340px !important;
    max-width: 420px !important;
    height: 650px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

#sidebar-tabs {
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    background: white !important;
    padding: 8px !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.05) !important;
}

#sidebar-tabs button {
    flex: 1 1 0 !important;
    border-radius: 12px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    min-height: 44px !important;
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
}

#sidebar-tabs button.active-tab {
    background: #fff7ed !important;
    color: #f97316 !important;
    border: 1px solid #fed7aa !important;
}

#sidebar-panel {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow-y: auto !important;
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 24px !important;
    padding: 24px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.05) !important;
}

#sidebar-panel * {
    color: #1e293b !important;
}

#sidebar-panel h3 {
    font-size: 1.35rem !important;
    margin-top: 0 !important;
    margin-bottom: 22px !important;
    padding-bottom: 14px !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

#sidebar-panel p {
    margin: 12px 0 !important;
    line-height: 1.6 !important;
    font-size: 1rem !important;
}

/* ===========================
   HUMAN APPROVAL
   =========================== */

#approval-row {
    gap: 12px !important;
    margin-top: 14px !important;
}

/* ===========================
   SESLİ YANIT
   =========================== */

#voice-toggle {
    margin-top: 16px !important;
}

#tts-output {
    margin-top: 12px !important;
    border-radius: 18px !important;
}

/* ===========================
   MESAJ ALANI
   =========================== */

#input-row {
    align-items: center !important;
    gap: 14px !important;
    margin-top: 14px !important;
}

#input-row textarea {
    min-height: 52px !important;
    color: #1e293b !important;
    background: white !important;
}

#send-btn button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    min-height: 52px !important;
    font-size: 1rem !important;
}

/* ===========================
   MİKROFON
   =========================== */

#mic-recorder {
    width: 100% !important;
    margin-top: 12px !important;
    border-radius: 18px !important;
}

/* Karşılıklı konuşmada TTS otomatik çalar; büyük oynatıcı gösterilmez. */
#conversation-tts-output {
    display: none !important;
}

#conversation-panel {
    margin-top: 18px !important;
}

/* Küçük ekran koruması */
@media (max-width: 900px) {
    #main-workspace {
        flex-direction: column !important;
    }

    #sidebar {
        min-width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
    }

    #sidebar-panel {
        max-height: 420px !important;
    }
}
"""

PLACEHOLDER_GRAPH_HTML = (
    "<div style='padding:20px;text-align:center;color:#64748b;'>"
    "Henüz bir görev çalıştırılmadı."
    "</div>"
)

PLACEHOLDER_AGENT_GRAPH_HTML = (
    "<div style='padding:30px;text-align:center;color:#64748b;'>"
    "Graph'ı görmek için butona bas."
    "</div>"
)

DEFAULT_TODO_TEXT = "### Yapılacaklar\nHenüz plan oluşturulmadı."


def select_tab(tab_name):
    """Sidebar sekmesi değiştiğinde ilgili paneli göster, diğerlerini gizle."""

    return (
        gr.update(visible=(tab_name == "todos")),
        gr.update(visible=(tab_name == "runtime")),
        gr.update(visible=(tab_name == "graph")),
        gr.update(elem_classes=["active-tab"] if tab_name == "todos" else []),
        gr.update(elem_classes=["active-tab"] if tab_name == "runtime" else []),
        gr.update(elem_classes=["active-tab"] if tab_name == "graph" else []),
    )


def set_conversation_layout(active):
    """Karşılıklı konuşma sırasında normal kontrolleri sade biçimde gizler."""
    return (
        gr.update(visible=not active),
        gr.update(visible=not active),
        gr.update(
            value=(
                "🟢 Dinliyorum… Konuşman bittiğinde otomatik cevap vereceğim."
                if active
                else "⚪ Karşılıklı konuşma durduruldu."
            )
        ),
    )


# ============================================================
# WEBRTC KARŞILIKLI KONUŞMA
# ============================================================

class AsyncAgentRunner:
    """Agentı WebRTC turları boyunca aynı asyncio döngüsünde tutar."""

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


webrtc_agent_runner = AsyncAgentRunner()
webrtc_turn_lock = threading.Lock()


def save_webrtc_input(audio):
    sample_rate, samples = audio
    samples = np.asarray(samples)

    if samples.ndim == 2:
        if samples.shape[0] <= 2:
            samples = samples.mean(axis=0)
        else:
            samples = samples.mean(axis=1)

    input_path = os.path.join(
        tempfile.gettempdir(),
        f"sidekick_webrtc_{uuid.uuid4().hex}.wav",
    )
    sf.write(input_path, np.squeeze(samples), int(sample_rate))
    return input_path


def yield_webrtc_audio(audio_path, frame_seconds=0.25):
    samples, sample_rate = sf.read(audio_path, dtype="float32")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    frame_size = max(int(sample_rate * frame_seconds), 1)

    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        if frame.size:
            yield int(sample_rate), frame.reshape(1, -1)


def webrtc_voice_reply(audio):
    """WebRTC -> faster-whisper -> Sidekick -> Chatterbox -> WebRTC."""
    with webrtc_turn_lock:
        total_started = time.perf_counter()
        input_path = save_webrtc_input(audio)

        try:
            stt_started = time.perf_counter()
            user_text = transcribe_audio(input_path)
            print(
                "[WEBRTC PERF] faster-whisper: "
                f"{time.perf_counter() - stt_started:.2f}s | "
                f'text="{user_text}"'
            )
        finally:
            try:
                os.remove(input_path)
            except OSError:
                pass

        if not user_text:
            return

        agent_started = time.perf_counter()
        answer, approval_needed, _, _ = webrtc_agent_runner.run(
            ask_agent(user_text, conversation_mode=True)
        )
        print(
            "[WEBRTC PERF] Agent: "
            f"{time.perf_counter() - agent_started:.2f}s"
        )

        if approval_needed:
            answer = (
                "Bu işlem kullanıcı onayı gerektiriyor. "
                "Normal Sidekick ekranından onaylayabilir veya reddedebilirsin."
            )

        tts_started = time.perf_counter()
        audio_path = synthesize_speech(answer)
        print(
            "[WEBRTC PERF] Chatterbox: "
            f"{time.perf_counter() - tts_started:.2f}s"
        )

        if not audio_path:
            return

        try:
            yield from yield_webrtc_audio(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        print(
            "[WEBRTC PERF] Toplam tur: "
            f"{time.perf_counter() - total_started:.2f}s"
        )


webrtc_stream = Stream(
    handler=ReplyOnPause(
        webrtc_voice_reply,
        can_interrupt=False,
    ),
    modality="audio",
    mode="send-receive",
)


with gr.Blocks(
    title="Sidekick",
    theme=THEME,
    css=CSS,
) as demo:

    with gr.Column(elem_id="header"):
        gr.Markdown("# Sidekick")
        gr.Markdown("Kişisel AI iş arkadaşın")

    with gr.Row(elem_id="main-workspace"):
        with gr.Column(scale=2, elem_id="chat-card"):
            chatbot = gr.Chatbot(
                label=None,
                show_label=False,
                type="messages",
                allow_tags=False,
                elem_id="chatbot",
            )

        with gr.Column(scale=1, elem_id="sidebar") as sidebar:

            with gr.Row(elem_id="sidebar-tabs"):
                tab_todos_btn = gr.Button(
                    "📋 Yapılacaklar",
                    elem_classes=["active-tab"],
                )
                tab_runtime_btn = gr.Button(
                    "🔀 Çalışma Akışı",
                )
                tab_graph_btn = gr.Button(
                    "🧩 Agent Graph",
                )

            with gr.Column(elem_id="sidebar-panel"):

                with gr.Column(visible=True) as todos_panel:
                    todo_panel = gr.Markdown(DEFAULT_TODO_TEXT)

                with gr.Column(visible=False) as runtime_panel:
                    runtime_graph_view = gr.HTML(PLACEHOLDER_GRAPH_HTML)

                with gr.Column(visible=False) as graph_panel:
                    gr.Markdown(
                        "Gerçek `create_agent()` / LangGraph çalışma yapısı."
                    )

                    graph_btn = gr.Button(
                        "Graph'ı Göster",
                        variant="secondary",
                    )

                    graph_view = gr.HTML(PLACEHOLDER_AGENT_GRAPH_HTML)

    # HUMAN APPROVAL
    with gr.Row(elem_id="approval-row"):
        approve_btn = gr.Button(
            "Onayla",
            variant="primary",
            visible=False,
        )

        reject_btn = gr.Button(
            "Reddet",
            visible=False,
        )

    # Yazılı mesaj ve tek seferlik mikrofon özellikleri normal modda kalır.
    # Karşılıklı konuşma kaydı başladığında bu grubun tamamı gizlenir.
    with gr.Column(visible=True) as normal_controls:
        voice_enabled = gr.Checkbox(
            label="🔊 Sesli yanıt",
            value=True,
        )

        audio_player = gr.Audio(
            label=None,
            show_label=False,
            autoplay=True,
            interactive=False,
            elem_id="tts-output",
        )

        with gr.Row(elem_id="input-row"):

            message = gr.Textbox(
                placeholder="Sidekick'e bir şey sor...",
                show_label=False,
                scale=8,
            )

            send = gr.Button(
                "Gönder",
                variant="primary",
                elem_id="send-btn",
                scale=1,
            )

        mic = gr.Audio(
            sources=["microphone"],
            type="filepath",
            show_label=False,
            editable=False,
            waveform_options=gr.WaveformOptions(
                waveform_color="#f97316",
                show_recording_waveform=False,
            ),
            elem_id="mic-recorder",
        )

    with gr.Column(elem_id="conversation-panel"):
        gr.Markdown("### Karşılıklı Konuşma Modu")
        gr.Markdown(
            "Kesintisiz sesli konuşma WebRTC ekranında çalışır. "
            "Normal mikrofon ve yazılı kullanım bu ekranda kullanılmaya devam eder."
        )
        gr.HTML(
            """
            <a
                href="http://127.0.0.1:7861"
                target="_blank"
                style="
                    display:inline-block;
                    padding:12px 18px;
                    border-radius:12px;
                    background:#f97316;
                    color:white;
                    font-weight:700;
                    text-decoration:none;
                "
            >
                Karşılıklı Konuşmayı Aç
            </a>
            """
        )


# ============================================================
# EVENTS
# ============================================================

    tab_todos_btn.click(
        lambda: select_tab("todos"),
        inputs=[],
        outputs=[
            todos_panel,
            runtime_panel,
            graph_panel,
            tab_todos_btn,
            tab_runtime_btn,
            tab_graph_btn,
        ],
    )

    tab_runtime_btn.click(
        lambda: select_tab("runtime"),
        inputs=[],
        outputs=[
            todos_panel,
            runtime_panel,
            graph_panel,
            tab_todos_btn,
            tab_runtime_btn,
            tab_graph_btn,
        ],
    )

    tab_graph_btn.click(
        lambda: select_tab("graph"),
        inputs=[],
        outputs=[
            todos_panel,
            runtime_panel,
            graph_panel,
            tab_todos_btn,
            tab_runtime_btn,
            tab_graph_btn,
        ],
    )

    graph_btn.click(
        get_agent_graph_html,
        inputs=[],
        outputs=[graph_view],
    )

    send.click(
        handle_text,
        inputs=[
            message,
            chatbot,
            voice_enabled,
        ],
        outputs=[
            chatbot,
            message,
            audio_player,
            approve_btn,
            reject_btn,
            todo_panel,
            runtime_graph_view,
        ],
    )

    message.submit(
        handle_text,
        inputs=[
            message,
            chatbot,
            voice_enabled,
        ],
        outputs=[
            chatbot,
            message,
            audio_player,
            approve_btn,
            reject_btn,
            todo_panel,
            runtime_graph_view,
        ],
    )

    mic.stop_recording(
        handle_audio,
        inputs=[
            mic,
            chatbot,
            voice_enabled,
        ],
        outputs=[
            chatbot,
            mic,
            audio_player,
            approve_btn,
            reject_btn,
            todo_panel,
            runtime_graph_view,
        ],
    )

    approve_btn.click(
        approve_action,
        inputs=[
            chatbot,
            voice_enabled,
        ],
        outputs=[
            chatbot,
            audio_player,
            approve_btn,
            reject_btn,
        ],
    )

    reject_btn.click(
        reject_action,
        inputs=[
            chatbot,
            voice_enabled,
        ],
        outputs=[
            chatbot,
            audio_player,
            approve_btn,
            reject_btn,
        ],
    )


if __name__ == "__main__":
    webrtc_stream.ui.launch(
        server_name="127.0.0.1",
        server_port=7861,
        prevent_thread_lock=True,
        quiet=True,
    )

    demo.launch()