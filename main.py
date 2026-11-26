import asyncio
import logging
import os
import random
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from faster_whisper import WhisperModel
import av
import numpy as np

logging.basicConfig(level=logging.INFO)

bot_token = os.environ["BOT_TOKEN"]

# tiny / base / small / medium / large-v3: чем больше, тем точнее, но медленнее и тяжелее
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
MAX_VOICE_DURATION_SECONDS = 180
MAX_REPLY_LENGTH = 3500  # лимит Telegram на сообщение 4096 символов

TOO_LONG_TEMPLATES = [
    "{nickname}: Столько я слушать не буду",
    "{nickname}: Это не войс, это подкаст. Пас",
]

EMPTY_TRANSCRIPT_TEMPLATES = [
    "{nickname}: Пиздит что-то непонятное",
    "{nickname}: 我今天去了一个同性恋俱乐部。",
    "{nickname}: Переводчик не справился, это не язык",
]

logging.info("Загружаю модель Whisper '%s' (при первом запуске она скачается)...", WHISPER_MODEL_SIZE)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
logging.info("Модель загружена")

dispatcher = Dispatcher()
# Распознаём по одному войсу за раз, чтобы не положить процессор
transcription_lock = asyncio.Lock()


def get_sender_nickname(message: Message) -> str:
    sender = message.from_user
    if sender is None:
        return "Аноним"
    # username без "@", чтобы бот не пинговал человека при каждом ответе
    return sender.username or sender.full_name


def decode_audio_to_array(audio_bytes: bytes) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    sample_chunks = []

    with av.open(BytesIO(audio_bytes), mode="r") as container:
        audio_stream = container.streams.audio[0]
        for frame in container.decode(audio_stream):
            for resampled_frame in resampler.resample(frame):
                sample_chunks.append(resampled_frame.to_ndarray().reshape(-1))
        for resampled_frame in resampler.resample(None):  # дочитать остаток
            sample_chunks.append(resampled_frame.to_ndarray().reshape(-1))

    if not sample_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(sample_chunks).astype(np.float32) / 32768.0


def transcribe_audio(audio_bytes: bytes) -> str:
    audio_samples = decode_audio_to_array(audio_bytes)
    if audio_samples.size == 0:
        return ""

    segments, _detected_info = whisper_model.transcribe(
        audio_samples,
        language="ru",
        vad_filter=True,
        beam_size=1,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


@dispatcher.message(F.voice)
async def handle_voice_message(message: Message, bot: Bot) -> None:
    sender_nickname = get_sender_nickname(message)
    voice = message.voice

    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.reply(random.choice(TOO_LONG_TEMPLATES).format(nickname=sender_nickname))
        return

    try:
        await bot.send_chat_action(message.chat.id, "typing")

        audio_buffer = BytesIO()
        await bot.download(voice, destination=audio_buffer)
        audio_bytes = audio_buffer.getvalue()

        async with transcription_lock:
            recognized_text = await asyncio.to_thread(transcribe_audio, audio_bytes)
    except Exception:
        logging.exception("Не удалось распознать голосовое сообщение")
        return

    if not recognized_text:
        reply_text = random.choice(EMPTY_TRANSCRIPT_TEMPLATES).format(nickname=sender_nickname)
    else:
        reply_text = f"{sender_nickname}: {recognized_text[:MAX_REPLY_LENGTH]}"

    await message.reply(reply_text)


async def main() -> None:
    bot = Bot(token=bot_token)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
