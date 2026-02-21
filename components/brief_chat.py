"""
Brief Chat Component - Reusable chat interface for Executive Brief Q&A.

Provides text-based chat with on-demand TTS (Read Aloud) for answers.
"""

import streamlit as st
from typing import Optional


def render_brief_chat(
    brief_content: str,
    history_key: str,
    llm_config: dict,
    voice_config: dict,
    expander_title: str = "Chat with your Brief"
) -> None:
    """
    Render a chat interface for executive brief Q&A.

    Args:
        brief_content: The executive brief text to use as context
        history_key: Session state key for chat history (e.g., "portfolio_brief_chat_history")
        llm_config: LLM configuration dict with provider, model, api_key, temperature, timeout
        voice_config: Voice configuration dict with voice_enabled, provider, use_llm_openai_key, openai_api_key, replicate_api_key
        expander_title: Title for the chat expander
    """
    # Import helper functions
    from utils.portfolio_settings import get_voice_openai_key, get_voice_replicate_key

    # Check if LLM is configured
    has_llm_key = bool(llm_config.get('api_key', '').strip()) if llm_config else False
    voice_enabled = voice_config.get('voice_enabled', False)

    # Determine voice provider and keys
    voice_provider = voice_config.get('provider', 'OpenAI')
    effective_openai_key = get_voice_openai_key(llm_config, voice_config)
    effective_replicate_key = get_voice_replicate_key(voice_config)

    # Check if we have any voice/TTS capability
    has_voice_openai = bool(effective_openai_key)
    has_voice_replicate = bool(effective_replicate_key)
    # For backwards compatibility with old config format
    if not has_voice_openai and not has_voice_replicate:
        # Check old format fields
        has_voice_replicate = bool(voice_config.get('replicate_api_key', '').strip())
        if has_voice_replicate:
            effective_replicate_key = voice_config.get('replicate_api_key', '')

    with st.expander(f"💬 {expander_title}", expanded=True):

        # Initialize chat history if not exists
        if history_key not in st.session_state:
            st.session_state[history_key] = []

        chat_history = st.session_state[history_key]

        # Display chat history with Read Aloud buttons
        if chat_history:
            for i, qa in enumerate(chat_history):
                with st.chat_message("user"):
                    st.write(qa['question'])
                with st.chat_message("assistant"):
                    # Escape $ signs to prevent LaTeX rendering issues
                    escaped_answer = qa['answer'].replace('$', '\\$')
                    st.markdown(escaped_answer)

                    # Read Aloud button (on-demand TTS) - show if any voice key is configured
                    has_any_voice_key = has_voice_openai or has_voice_replicate
                    if has_any_voice_key:
                        col_tts, col_spacer = st.columns([1, 4])
                        with col_tts:
                            tts_key = f"tts_{history_key}_{i}"
                            if st.button("🔊 Read Aloud", key=tts_key, help="Generate audio for this answer"):
                                _synthesize_and_play(qa['answer'], effective_openai_key, effective_replicate_key, tts_key)

                        # Display audio if already generated
                        audio_key = f"audio_{history_key}_{i}"
                        if audio_key in st.session_state and st.session_state[audio_key]:
                            # Auto-detect format: OpenAI returns MP3, Replicate returns WAV
                            audio_data = st.session_state[audio_key]
                            audio_format = 'audio/mp3' if audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb' else 'audio/wav'
                            st.audio(audio_data, format=audio_format)
        else:
            st.caption("Ask questions about the executive brief. Your conversation history will appear here.")

        st.markdown("---")

        # PRIMARY: Text chat input
        st.markdown("##### Ask a question")

        if not has_llm_key:
            st.warning("⚠️ LLM API key required to chat. Configure in Portfolio Management settings.")
        else:
            # Use form for text input with clear on submit
            with st.form(key=f"chat_form_{history_key}", clear_on_submit=True):
                user_question = st.text_input(
                    "Type your question about the brief...",
                    placeholder="e.g., What are the main risks identified?",
                    label_visibility="collapsed"
                )
                submit = st.form_submit_button("💬 Ask", type="primary", width='stretch')

                if submit and user_question.strip():
                    _process_text_question(
                        question=user_question.strip(),
                        brief_content=brief_content,
                        history_key=history_key,
                        llm_config=llm_config
                    )

        # SECONDARY: Voice input (if voice is enabled and any voice key is configured for ASR)
        voice_available = voice_enabled and (has_voice_openai or has_voice_replicate)

        if voice_available:
            st.markdown("---")
            st.markdown("##### Or use voice")
            if voice_provider == 'OpenAI' and has_voice_openai:
                st.caption("Using OpenAI Whisper for speech recognition")
            elif voice_provider == 'Replicate' and has_voice_replicate:
                st.caption("Using Replicate for speech recognition")

            # Session state key for pending transcribed question
            pending_key = f"pending_voice_question_{history_key}"

            # Check if we have a pending transcribed question to ask
            if pending_key in st.session_state and st.session_state[pending_key]:
                transcribed_question = st.session_state[pending_key]
                st.info(f"📝 **Transcribed question:** {transcribed_question}")

                col_ask, col_clear = st.columns([1, 1])
                with col_ask:
                    if st.button("💬 Ask this question", key=f"ask_transcribed_{history_key}", type="primary"):
                        if not has_llm_key:
                            st.error("❌ LLM API key required for answers. Configure in Portfolio Management.")
                        else:
                            # Process the transcribed question
                            _process_text_question(
                                question=transcribed_question,
                                brief_content=brief_content,
                                history_key=history_key,
                                llm_config=llm_config
                            )
                            # Clear pending question (will happen after rerun from _process_text_question)
                            st.session_state[pending_key] = ""
                with col_clear:
                    if st.button("🗑️ Discard", key=f"discard_transcribed_{history_key}"):
                        st.session_state[pending_key] = ""
                        st.rerun()
            else:
                # Show audio input for recording
                audio_input = st.audio_input(
                    "Record your question",
                    key=f"voice_input_{history_key}"
                )

                if audio_input is not None:
                    if st.button("🎤 Transcribe", key=f"transcribe_voice_{history_key}"):
                        _transcribe_voice_question(
                            audio_bytes=audio_input.getvalue(),
                            pending_key=pending_key,
                            openai_key=effective_openai_key,
                            replicate_key=effective_replicate_key
                        )

        # Clear history button
        if chat_history:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", key=f"clear_{history_key}"):
                st.session_state[history_key] = []
                # Clear any cached audio
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith(f"audio_{history_key}")]
                for k in keys_to_clear:
                    del st.session_state[k]
                st.rerun()


def _process_text_question(
    question: str,
    brief_content: str,
    history_key: str,
    llm_config: dict
) -> None:
    """Process a text question and add to chat history."""
    try:
        from services.voice_qa_service import VoiceQAService

        # We need a dummy Replicate key to instantiate, but we'll only use generate_answer
        # Create a minimal service instance just for the LLM call
        with st.spinner("Generating answer..."):
            answer = _call_llm_directly(
                question=question,
                brief_content=brief_content,
                chat_history=st.session_state[history_key],
                llm_config=llm_config
            )

        # Append to history
        st.session_state[history_key].append({
            'question': question,
            'answer': answer
        })
        st.rerun()

    except Exception as e:
        st.error(f"Error generating answer: {e}")


def _transcribe_voice_question(
    audio_bytes: bytes,
    pending_key: str,
    openai_key: str,
    replicate_key: str
) -> None:
    """Transcribe voice to text and store in session state for user review."""
    try:
        from services.voice_qa_service import VoiceQAService

        # Ensure we have at least one ASR service
        if not openai_key and not replicate_key:
            st.error("Voice input requires either OpenAI API key or Replicate API key")
            return

        service = VoiceQAService(
            replicate_api_key=replicate_key,
            openai_api_key=openai_key
        )

        with st.spinner("Transcribing audio..."):
            question = service.transcribe_audio(audio_bytes)

        # Store transcribed question for user to review and confirm
        st.session_state[pending_key] = question
        st.rerun()

    except Exception as e:
        st.error(f"Transcription error: {e}")


def _synthesize_and_play(text: str, openai_key: str, replicate_key: str, key_prefix: str) -> None:
    """Synthesize speech for text and store in session state."""
    try:
        from services.voice_qa_service import VoiceQAService

        with st.spinner("Generating audio..."):
            service = VoiceQAService(
                replicate_api_key=replicate_key,
                openai_api_key=openai_key
            )
            audio_bytes, error_msg = service.synthesize_speech(text)

            if audio_bytes:
                # Store in session state so it persists
                audio_key = key_prefix.replace("tts_", "audio_")
                st.session_state[audio_key] = audio_bytes
                st.rerun()
            else:
                st.error(f"Failed to generate audio: {error_msg or 'Unknown error'}")

    except Exception as e:
        st.error(f"TTS error: {type(e).__name__}: {e}")


def _call_llm_directly(
    question: str,
    brief_content: str,
    chat_history: list,
    llm_config: dict
) -> str:
    """Call LLM directly without needing Replicate API key."""
    import requests

    provider = llm_config.get("provider", "")
    model = llm_config.get("model", "")
    api_key = llm_config.get("api_key", "")
    temperature = float(llm_config.get("temperature", 0.2))
    timeout_sec = int(llm_config.get("timeout", 60))

    if not api_key.strip():
        raise ValueError("LLM API key is not configured")

    system_prompt = (
        "You are a project controls expert specializing in earned value management. "
        "Answer concisely in under 200 words based ONLY on the report context provided. "
        "Use specific numbers and project names from the report. "
        "IMPORTANT: Always use full names for EVM terms, not abbreviations. "
        "Examples: say 'Actual Cost' not 'AC', 'Budget at Completion' not 'BAC', "
        "'Planned Value' not 'PV', 'Earned Value' not 'EV', "
        "'Schedule Performance Index' not 'SPI', 'Cost Performance Index' not 'CPI', "
        "'Estimate to Complete' not 'ETC', 'Estimate at Completion' not 'EAC', "
        "'Original Duration' not 'OD', 'To-Complete Performance Index' not 'TCPI', "
        "'Variance at Completion' not 'VAC', 'Cost Variance' not 'CV', 'Schedule Variance' not 'SV'."
    )

    # Build conversation messages for follow-up context
    history_messages = []
    for qa in chat_history:
        history_messages.append({"role": "user", "content": qa["question"]})
        history_messages.append({"role": "assistant", "content": qa["answer"]})

    user_content = (
        f"EXECUTIVE BRIEF REPORT:\n{brief_content}\n\n"
        f"QUESTION: {question}"
    )

    if provider == "OpenAI":
        return _call_openai(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
    elif provider == "Gemini":
        return _call_gemini(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
    elif provider == "Claude":
        return _call_claude(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
    elif provider == "Kimi":
        return _call_kimi(api_key, model, system_prompt, history_messages, user_content, temperature, timeout_sec)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _call_openai(api_key, model, system_prompt, history, user_content, temperature, timeout):
    import requests
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


def _call_gemini(api_key, model, system_prompt, history, user_content, temperature, timeout):
    import requests
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


def _call_claude(api_key, model, system_prompt, history, user_content, temperature, timeout):
    import requests
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


def _call_kimi(api_key, model, system_prompt, history, user_content, temperature, timeout):
    import requests
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


def clear_chat_history(history_key: str) -> None:
    """Clear chat history for a given key. Call this when generating a new brief."""
    if history_key in st.session_state:
        st.session_state[history_key] = []

    # Clear any cached audio
    keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith(f"audio_{history_key}")]
    for k in keys_to_clear:
        del st.session_state[k]

    # Clear any pending voice question
    pending_key = f"pending_voice_question_{history_key}"
    if pending_key in st.session_state:
        st.session_state[pending_key] = ""
