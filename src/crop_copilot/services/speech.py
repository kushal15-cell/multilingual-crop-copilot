from __future__ import annotations

from pathlib import Path


class LocalWhisperSTT:
    def __init__(self, model_size: str = "base") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install speech support with: pip install -e ".[speech]"') from exc
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
        )

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        segments, _ = self.model.transcribe(str(audio_path), language=language, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ValueError("No speech was detected. Record again closer to the microphone.")
        return text


class GTTSService:
    def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError('Install speech support with: pip install -e ".[speech]"') from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gTTS(text=text, lang=language).save(str(output_path))
        return output_path
