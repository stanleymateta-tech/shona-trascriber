import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Project Nyaradzai — Shona Transcriber",
    page_icon="🎙",
    layout="centered"
)

st.title("🎙 Project Nyaradzai — Shona Transcriber 🇿🇼")
st.markdown("""
**The first open Shona (ChiShona) speech transcription tool.**
Upload any audio or video file and get a Shona transcript.
Your corrections help train the next, more accurate version.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

AWS_SERVER = None
SUPPORTED_TYPES = [
    "wav", "flac", "ogg", "mp3", "m4a", "mp4",
    "mov", "avi", "mkv", "webm", "aac", "opus"
]

# ── audio extraction ──────────────────────────────────────────────────────────
def get_audio_array(audio_bytes, suffix):
    import numpy as np
    import tempfile, os
    import soundfile as sf

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_in = f.name
    tmp_wav = tmp_in.replace(suffix, "_converted.wav")

    try:
        try:
            audio_array, sr = sf.read(tmp_in)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(
                    audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
            return audio_array.astype(np.float32)
        except Exception:
            pass

        import subprocess
        result = subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in,
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav
        ], capture_output=True)
        if result.returncode == 0 and os.path.exists(tmp_wav):
            audio_array, sr = sf.read(tmp_wav)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32)

        import librosa
        audio_array, _ = librosa.load(tmp_in, sr=16000, mono=True)
        return audio_array.astype(np.float32)

    finally:
        for p in [tmp_in, tmp_wav]:
            try:
                if os.path.exists(p): os.unlink(p)
            except Exception:
                pass

# ── transcription ─────────────────────────────────────────────────────────────
def transcribe_local(audio_array):
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model="Starsm91/whisper-small-shona",
        generate_kwargs={"language": "shona", "task": "transcribe"},
    )
    result = asr(
        audio_array,
        return_timestamps=True,
        generate_kwargs={"language": "shona", "task": "transcribe"},
    )
    return result["text"].strip()

# ── feedback storage ──────────────────────────────────────────────────────────
def save_correction(audio_bytes, suffix, original_text, corrected_text, filename):
    """
    Save a user correction to the Hugging Face dataset repository.
    Each correction = audio file + original transcript + corrected transcript.
    These accumulate as training data for the next model version.
    """
    try:
        from huggingface_hub import HfApi
        import tempfile, os, json
        from datetime import datetime

        api = HfApi()
        repo_id = "Starsm91/shona-corrections"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        # Save audio
        audio_filename = f"audio/{timestamp}{suffix}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_audio = f.name

        api.upload_file(
            path_or_fileobj=tmp_audio,
            path_in_repo=audio_filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        os.unlink(tmp_audio)

        # Save metadata
        meta = {
            "timestamp": timestamp,
            "filename": filename,
            "audio_path": audio_filename,
            "original": original_text,
            "corrected": corrected_text,
        }
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        meta_filename = f"corrections/{timestamp}.json"
        api.upload_file(
            path_or_fileobj=meta_json.encode(),
            path_in_repo=meta_filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return True
    except Exception as e:
        st.warning(f"Could not save correction: {e}")
        return False

# ── UI ────────────────────────────────────────────────────────────────────────
if AWS_SERVER:
    st.success("⚡ Connected to AWS GPU server")
else:
    st.info("ℹ️ Using local CPU — transcription takes 1-2 minutes per file")

st.subheader("Upload a file")
st.caption("WAV · FLAC · OGG · MP3 · M4A · AAC · MP4 · MOV · AVI · MKV · WEBM · OPUS")

uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    audio_bytes = uploaded_file.read()

    if suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    if st.button("🎙 Transcribe", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                audio_array = get_audio_array(audio_bytes, suffix)
                text = transcribe_local(audio_array)
                st.session_state["transcript"] = text
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["suffix"] = suffix
                st.session_state["filename"] = uploaded_file.name
            except Exception as e:
                st.error(f"Could not process file: {str(e)}")

# Show transcript and feedback section
if "transcript" in st.session_state and st.session_state["transcript"]:
    text = st.session_state["transcript"]

    st.success("Transcription complete!")
    st.subheader("Transcript")

    # Editable text area — user can correct mistakes
    corrected = st.text_area(
        "Review and correct the transcript if needed:",
        value=text,
        height=200,
        help="Edit any mistakes you see. Your corrections help train the next version."
    )

    # Download buttons
    col1, col2 = st.columns(2)
    base_name = st.session_state["filename"].rsplit(".", 1)[0]
    with col1:
        st.download_button(
            label="📄 Download as .txt",
            data=corrected,
            file_name=base_name + "_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        srt = f"1\n00:00:00,000 --> 00:05:00,000\n{corrected}\n"
        st.download_button(
            label="🎬 Download as .srt",
            data=srt,
            file_name=base_name + ".srt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    # Feedback section
    st.subheader("Help improve the model")
    st.markdown("""
    Did you correct any mistakes above? Submit your correction to help train
    a more accurate Shona model. Your audio and corrected text will be saved
    as open training data for the next version.
    """)

    accuracy = st.radio(
        "How accurate was the original transcription?",
        ["Excellent — no corrections needed",
         "Good — minor corrections",
         "Fair — several corrections",
         "Poor — many corrections"],
        horizontal=True,
    )

    if st.button("✅ Submit correction", type="secondary", use_container_width=True):
        with st.spinner("Saving your contribution..."):
            saved = save_correction(
                audio_bytes=st.session_state["audio_bytes"],
                suffix=st.session_state["suffix"],
                original_text=text,
                corrected_text=corrected,
                filename=st.session_state["filename"],
            )
        if saved:
            st.success(
                "Tatenda! Your correction has been saved. "
                "Every contribution makes the next Shona model more accurate."
            )
        else:
            st.info(
                "Correction could not be saved automatically, but you can "
                "contribute directly at: github.com/stanleymateta-tech/Project-Nyaradzai"
            )

st.divider()
st.markdown("""
### How the feedback loop works:
1. You upload Shona audio → model transcribes it
2. You correct any mistakes in the text box above
3. Submit → your corrected audio+text saves as training data
4. We periodically retrain the model on all corrections
5. The model gets more accurate on Zimbabwean speech over time

### About:
- GitHub: [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)
- Model: [Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona)
- Community: [Masakhane](https://masakhane.io)
""")
