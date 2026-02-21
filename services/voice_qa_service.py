"""
Voice Q&A Service for Portfolio Executive Briefs.

Hybrid pipeline:
  1. ASR:    Kimi-Audio-7B-Instruct on Replicate (audio -> text) OR OpenAI Whisper
  2. Answer: User's configured text LLM (GPT-4o / Claude / Gemini / Kimi text)
  3. TTS:   OpenAI TTS (preferred) OR Kimi-Audio on Replicate (fallback)
"""

import io
import json
import struct
import requests
from typing import Optional, Tuple, List


class VoiceQAService:
    """Orchestrates voice-based Q&A over executive briefing reports."""

    # Replicate models - using faster, more reliable alternatives
    REPLICATE_WHISPER = "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
    REPLICATE_TTS = "cjwbw/parler-tts:e0e2f9a22436df485bbaf846d0750619f47dd74058eb4307c37b5cd0dc80d09e"

    def __init__(self, replicate_api_key: str = "", openai_api_key: str = ""):
        self.replicate_api_key = replicate_api_key.strip() if replicate_api_key else ""
        self.openai_api_key = openai_api_key.strip() if openai_api_key else ""
        if not self.replicate_api_key and not self.openai_api_key:
            raise ValueError("Either Replicate or OpenAI API key is required")

    # ------------------------------------------------------------------
    # Step 1 – ASR (Speech-to-Text)
    # ------------------------------------------------------------------
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text. Prefers OpenAI Whisper, falls back to Replicate."""
        # Try OpenAI Whisper first (faster and more reliable)
        if self.openai_api_key:
            try:
                return self._transcribe_openai(audio_bytes)
            except Exception as e:
                # If OpenAI fails and we have Replicate, try that
                if self.replicate_api_key:
                    pass  # Fall through to Replicate
                else:
                    raise RuntimeError(f"OpenAI ASR failed: {type(e).__name__}: {e}")

        # Fall back to Replicate
        if not self.replicate_api_key:
            raise RuntimeError("No ASR service configured (need OpenAI or Replicate API key)")

        return self._transcribe_replicate(audio_bytes)

    def _transcribe_openai(self, audio_bytes: bytes) -> str:
        """Transcribe using OpenAI Whisper API."""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"

        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                files={"file": ("recording.wav", audio_file, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI Whisper failed: {type(e).__name__}: {e}")

    def _transcribe_replicate(self, audio_bytes: bytes) -> str:
        """Transcribe using OpenAI Whisper on Replicate."""
        import replicate
        import base64

        client = replicate.Client(api_token=self.replicate_api_key)

        # Convert audio to base64 data URI for Whisper model
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_uri = f"data:audio/wav;base64,{audio_b64}"

        try:
            output = client.run(
                self.REPLICATE_WHISPER,
                input={
                    "audio": audio_uri,
                    "model": "large-v3",
                    "language": "en",
                    "translate": False,
                    "temperature": 0,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Replicate Whisper failed: {type(e).__name__}: {e}")

        # Whisper returns a dict with 'transcription' key
        if isinstance(output, dict):
            return output.get("transcription", "").strip()
        if isinstance(output, str):
            return output.strip()
        # Handle iterator
        try:
            text = "".join(str(chunk) for chunk in output)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse Whisper output: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Step 2 – Text LLM answer generation
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        question: str,
        report_context: str,
        conversation_history: List[dict],
        llm_config: dict,
    ) -> str:
        """Generate a text answer using the user's configured LLM provider."""

        provider = llm_config.get("provider", "")
        model = llm_config.get("model", "")
        api_key = llm_config.get("api_key", "")
        temperature = float(llm_config.get("temperature", 0.2))
        timeout_sec = int(llm_config.get("timeout", 60))

        if not api_key.strip():
            raise ValueError("Text LLM API key is not configured")

        system_prompt = (
            "You are a project controls expert specializing in earned value management. "
            "Answer concisely in under 200 words based ONLY on the report context provided. "
            "Use specific numbers and project names from the report."
        )

        # Build conversation messages for follow-up context
        history_messages = []
        for qa in conversation_history:
            history_messages.append({"role": "user", "content": qa["question"]})
            history_messages.append({"role": "assistant", "content": qa["answer"]})

        user_content = (
            f"EXECUTIVE BRIEF REPORT:\n{report_context}\n\n"
            f"QUESTION: {question}"
        )

        if provider == "OpenAI":
            return self._call_openai(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
        elif provider == "Gemini":
            return self._call_gemini(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
        elif provider == "Claude":
            return self._call_claude(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
        elif provider == "Kimi":
            return self._call_kimi(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # ------------------------------------------------------------------
    # Step 3 – TTS (Text-to-Speech)
    # ------------------------------------------------------------------
    def synthesize_speech(self, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Convert text to speech. Prefers OpenAI TTS, falls back to Replicate.
        Returns (audio bytes or None, error message or None)."""
        # Try OpenAI TTS first (faster and more reliable)
        if self.openai_api_key:
            audio, error = self._tts_openai(text)
            if audio:
                return audio, None
            # If OpenAI fails and we have Replicate, try that
            if self.replicate_api_key:
                openai_error = error  # Save for combined error message
            else:
                return None, error

        # Fall back to Replicate
        if not self.replicate_api_key:
            return None, "No TTS service configured (need OpenAI or Replicate API key)"

        audio, error = self._tts_replicate(text)
        if audio:
            return audio, None

        # If both failed, combine error messages
        if self.openai_api_key:
            return None, f"OpenAI TTS failed: {openai_error}; Replicate TTS failed: {error}"
        return None, error

    def _tts_openai(self, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """TTS using OpenAI API. Returns (audio bytes, error message)."""
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": "alloy",
                    "response_format": "mp3",
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.content, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _tts_replicate(self, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """TTS using Parler-TTS on Replicate. Returns (audio bytes, error message)."""
        try:
            import replicate

            client = replicate.Client(api_token=self.replicate_api_key)

            try:
                output = client.run(
                    self.REPLICATE_TTS,
                    input={
                        "prompt": text,
                        "description": "A male speaker with a clear, professional voice delivers the text at a moderate pace in a neutral American accent.",
                    },
                )
            except Exception as e:
                return None, f"Replicate Parler-TTS failed: {type(e).__name__}: {e}"

            # Handle URL string response
            if isinstance(output, str):
                try:
                    resp = requests.get(output, timeout=120)
                    resp.raise_for_status()
                    return resp.content, None
                except Exception as e:
                    return None, f"Failed to download audio: {type(e).__name__}: {e}"

            # Handle FileOutput object (has .url or can be read)
            if hasattr(output, 'url'):
                try:
                    resp = requests.get(output.url, timeout=120)
                    resp.raise_for_status()
                    return resp.content, None
                except Exception as e:
                    return None, f"Failed to download audio: {type(e).__name__}: {e}"

            if hasattr(output, "read"):
                return output.read(), None

            return None, f"Unexpected Parler-TTS output type: {type(output)}"

        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process_question(
        self,
        audio_bytes: bytes,
        report_context: str,
        conversation_history: List[dict],
        llm_config: dict,
    ) -> Tuple[str, str, Optional[bytes], Optional[str]]:
        """Run the full ASR -> LLM -> TTS pipeline.
        Returns (question_text, answer_text, audio_bytes_or_none, tts_error_or_none).
        """
        question = self.transcribe_audio(audio_bytes)
        answer = self.generate_answer(question, report_context, conversation_history, llm_config)
        audio, tts_error = self.synthesize_speech(answer)
        return question, answer, audio, tts_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_silent_wav(self, duration_ms: int = 100) -> io.BytesIO:
        """Create a minimal silent WAV file for TTS input."""
        sample_rate = 16000
        num_samples = int(sample_rate * duration_ms / 1000)
        data_size = num_samples * 2  # 16-bit mono
        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVEfmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(b"\x00" * data_size)
        buf.seek(0)
        return buf

    # -- Provider-specific LLM calls --

    def _call_openai(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json={"model": model.strip(), "messages": messages, "temperature": temperature, "max_tokens": 2000},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        model_name = model.strip()
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        # Build multi-turn contents
        contents = []
        full_first = f"{system_prompt}\n\n{history[0]['content']}" if history else f"{system_prompt}\n\n{user_content}"
        if history:
            contents.append({"role": "user", "parts": [{"text": full_first}]})
            for msg in history[1:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            contents.append({"role": "user", "parts": [{"text": user_content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": full_first}]})

        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key.strip()},
            json={"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": 2000}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_claude(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        messages = list(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key.strip(), "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": model.strip(), "max_tokens": 2000, "temperature": temperature, "system": system_prompt, "messages": messages},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        return text.strip()

    def _call_kimi(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.moonshot.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json={"model": model.strip(), "messages": messages, "temperature": temperature, "max_tokens": 2000},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
