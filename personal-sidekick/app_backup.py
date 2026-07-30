import os
import uuid
import tempfile

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.types import Command

from sidekick.agent import build_agent

load_dotenv()

client = OpenAI()

agent = None
thread_id = str(uuid.uuid4())
pending_interrupt = False


# ============================================================
# AGENT
# ============================================================

async def initialize():
    global agent

    if agent is None:
        agent = await build_agent()


async def ask_agent(message: str):
    global agent, pending_interrupt

    await initialize()

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
                    "content": message
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

        return text, True

    pending_interrupt = False

    return result["messages"][-1].content, False


# ============================================================
# SPEECH
# ============================================================

def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )

    return transcript.text.strip()


def synthesize_speech(text: str):
    if not text:
        return None

    speech_path = os.path.join(
        tempfile.gettempdir(),
        f"sidekick_{uuid.uuid4().hex}.mp3"
    )

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    response.stream_to_file(speech_path)

    return speech_path


# ============================================================
# CHAT
# ============================================================

async def handle_message(message, history, voice_enabled):
    history = history or []

    if not message or not message.strip():
        return (
            history,
            "",
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    answer, approval_needed = await ask_agent(message)

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
        audio_out = synthesize_speech(answer)

    return (
        history,
        "",
        audio_out,
        gr.update(visible=approval_needed),
        gr.update(visible=approval_needed),
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
        )

    user_text = transcribe_audio(audio_path)

    result = await handle_message(
        user_text,
        history,
        voice_enabled
    )

    history, _, audio_out, approve_update, reject_update = result

    return (
        history,
        None,
        audio_out,
        approve_update,
        reject_update,
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
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

#header {
    text-align: center;
    padding: 30px 0 8px 0;
}

#header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f97316, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

#chatbot {
    min-height: 500px;
}

#input-row {
    align-items: center;
    gap: 10px;
}

#send-btn button {
    border-radius: 14px !important;
    font-weight: 600;
}

#approval-row {
    gap: 10px;
}

#mic-recorder {
    width: 100% !important;
}

#tts-output {
    margin-top: 4px;
}
"""


with gr.Blocks(title="Sidekick") as demo:

    with gr.Column(elem_id="header"):
        gr.Markdown("# Sidekick")
        gr.Markdown("Kişisel AI iş arkadaşın")

    chatbot = gr.Chatbot(
        label=None,
        show_label=False,
        elem_id="chatbot",
    )

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


# ============================================================
# EVENTS
# ============================================================

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
    demo.launch(
        theme=THEME,
        css=CSS,
    )
