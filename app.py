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
**The first open Shona speech transcription tool — now with speaker detection.**
Upload audio or video. Get a Shona transcript with each speaker labelled.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

HF_TOKEN  = st.secrets.get("HF_TOKEN", None)
AWS_SERVER = None
SUPPORTED_TYPES = [
    "wav","flac","ogg","mp3","m4a","mp4","mov","avi","mkv","webm","aac","opus"
]

# ── audio extraction ──────────────────────────────────────────────────────────
def get_audio_array_and_wav(audio_bytes, suffix):
    """Return (numpy array, wav_path) — wav file needed for diarisation."""
    import numpy as np, tempfile, os, soundfile as sf

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes); tmp_in = f.name
    tmp_wav = tempfile.mktemp(suffix=".wav")

    try:
        import subprocess
        subprocess.run(
            ["ffmpeg","-y","-i",tmp_in,"-ar","16000","-ac","1","-f","wav",tmp_wav],
            capture_output=True)
        if os.path.exists(tmp_wav):
            arr, _ = sf.read(tmp_wav)
            return arr.astype(np.float32), tmp_wav
    except Exception:
        pass

    try:
        arr, sr = sf.read(tmp_in)
        if len(arr.shape) > 1: arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
        sf.write(tmp_wav, arr, 16000)
        return arr.astype(np.float32), tmp_wav
    finally:
        try: os.unlink(tmp_in)
        except: pass

# ── noise suppression ─────────────────────────────────────────────────────────
def suppress_noise(audio_array):
    """Basic spectral noise gate using noisereduce."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio_array, sr=16000, stationary=False)
    except Exception:
        return audio_array   # return original if noisereduce not available

# ── speaker diarisation ───────────────────────────────────────────────────────
def diarise(wav_path, token):
    """
    Run pyannote speaker diarisation.
    Returns list of (start, end, speaker_label) tuples.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        output = pipeline(wav_path)
        segments = []
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return segments
    except Exception as e:
        return None

# ── transcription ─────────────────────────────────────────────────────────────
def transcribe_segment(asr, audio_array, start, end, sr=16000):
    """Transcribe a specific time segment."""
    s = int(start * sr)
    e = int(end   * sr)
    segment = audio_array[s:e]
    if len(segment) < 1600:   # less than 0.1s — skip
        return ""
    try:
        result = asr(segment.astype(float),
                     generate_kwargs={"language":"shona","task":"transcribe"})
        return result["text"].strip()
    except Exception:
        return ""

def transcribe_full(audio_array):
    """Transcribe the whole file at once."""
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="Starsm91/whisper-small-shona",
                   generate_kwargs={"language":"shona","task":"transcribe"})
    result = asr(audio_array, return_timestamps=True,
                 generate_kwargs={"language":"shona","task":"transcribe"})
    return result["text"].strip()

def transcribe_with_speakers(audio_array, segments):
    """Transcribe each speaker segment separately."""
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="Starsm91/whisper-small-shona",
                   generate_kwargs={"language":"shona","task":"transcribe"})
    lines = []
    for start, end, speaker in segments:
        text = transcribe_segment(asr, audio_array, start, end)
        if text:
            mins  = int(start // 60)
            secs  = int(start % 60)
            label = speaker.replace("SPEAKER_","Speaker ")
            lines.append(f"[{mins:02d}:{secs:02d}] {label}: {text}")
    return "\n\n".join(lines)

def make_srt(segments_with_text):
    """Generate SRT subtitle file from speaker segments."""
    srt = []
    for i, (start, end, speaker, text) in enumerate(segments_with_text, 1):
        def fmt(t):
            h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t%1)*1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        label = speaker.replace("SPEAKER_","Speaker ")
        srt.append(f"{i}\n{fmt(start)} --> {fmt(end)}\n[{label}] {text}\n")
    return "\n".join(srt)

# ── feedback storage ──────────────────────────────────────────────────────────
def save_correction(audio_bytes, suffix, original_text, corrected_text, filename):
    try:
        from huggingface_hub import HfApi
        import tempfile, os, json
        from datetime import datetime
        api   = HfApi()
        ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        rid   = "Starsm91/shona-corrections"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes); tmp = f.name
        api.upload_file(path_or_fileobj=tmp,
                        path_in_repo=f"audio/{ts}{suffix}",
                        repo_id=rid, repo_type="dataset")
        os.unlink(tmp)
        meta = json.dumps({"timestamp":ts,"filename":filename,
                           "original":original_text,"corrected":corrected_text},
                          ensure_ascii=False)
        api.upload_file(path_or_fileobj=meta.encode(),
                        path_in_repo=f"corrections/{ts}.json",
                        repo_id=rid, repo_type="dataset")
        return True
    except Exception as e:
        return False

# ── sidebar options ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    use_diarisation = st.toggle(
        "👥 Identify speakers",
        value=False,
        help="Label who spoke when. Requires a Hugging Face token with pyannote access."
    )
    use_noise_reduction = st.toggle(
        "🔇 Reduce background noise",
        value=True,
        help="Filter out background noise before transcribing."
    )
    if use_diarisation:
        if HF_TOKEN:
            st.success("Hugging Face token found")
        else:
            user_token = st.text_input(
                "Hugging Face token (for speaker ID)",
                type="password",
                help="Get a Write token from huggingface.co/settings/tokens. "
                     "You must also accept terms at "
                     "hf.co/pyannote/speaker-diarization-community-1"
            )
            if user_token:
                HF_TOKEN = user_token

    st.divider()
    st.markdown("**About:**")
    st.markdown("[Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)")
    st.markdown("[Shona ASR Model](https://huggingface.co/Starsm91/whisper-small-shona)")

# ── main UI ───────────────────────────────────────────────────────────────────
st.subheader("Upload a file")
st.caption("WAV · FLAC · OGG · MP3 · M4A · AAC · MP4 · MOV · AVI · MKV · WEBM · OPUS")

uploaded_file = st.file_uploader(
    "Choose an audio or video file", type=SUPPORTED_TYPES)

if uploaded_file is not None:
    suffix     = "." + uploaded_file.name.split(".")[-1].lower()
    audio_bytes = uploaded_file.read()

    if suffix in [".mp4",".mov",".avi",".mkv",".webm"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    if st.button("🎙 Transcribe", type="primary", use_container_width=True):
        import os

        with st.spinner("Extracting audio..."):
            audio_array, wav_path = get_audio_array_and_wav(audio_bytes, suffix)

        if use_noise_reduction:
            with st.spinner("Reducing background noise..."):
                audio_array = suppress_noise(audio_array)

        transcript = ""
        segments_with_text = []

        if use_diarisation and HF_TOKEN:
            with st.spinner("Identifying speakers... (this takes a minute)"):
                segments = diarise(wav_path, HF_TOKEN)

            if segments:
                with st.spinner("Transcribing each speaker..."):
                    transcript = transcribe_with_speakers(audio_array, segments)
                    # also build segments_with_text for SRT
                    import transformers
                    transformers.logging.set_verbosity_error()
                    from transformers import pipeline as hf_pipeline
                    asr = hf_pipeline("automatic-speech-recognition",
                                      model="Starsm91/whisper-small-shona",
                                      generate_kwargs={"language":"shona","task":"transcribe"})
                    for start, end, speaker in segments:
                        text = transcribe_segment(asr, audio_array, start, end)
                        if text:
                            segments_with_text.append((start, end, speaker, text))
            else:
                st.warning("Speaker identification failed — transcribing without speaker labels.")
                with st.spinner("Transcribing..."):
                    transcript = transcribe_full(audio_array)
        else:
            with st.spinner("Transcribing Shona... (1-2 minutes)"):
                transcript = transcribe_full(audio_array)

        try:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception:
            pass

        st.session_state["transcript"]  = transcript
        st.session_state["audio_bytes"] = audio_bytes
        st.session_state["suffix"]      = suffix
        st.session_state["filename"]    = uploaded_file.name
        st.session_state["srt_data"]    = make_srt(segments_with_text) if segments_with_text else None

# ── results ───────────────────────────────────────────────────────────────────
if st.session_state.get("transcript"):
    transcript = st.session_state["transcript"]
    st.success("Transcription complete!")
    st.subheader("Transcript")

    corrected = st.text_area(
        "Review and correct the transcript if needed:",
        value=transcript, height=250,
        help="Edit mistakes — your corrections help train the next version")

    col1, col2 = st.columns(2)
    base = st.session_state["filename"].rsplit(".",1)[0]
    with col1:
        st.download_button("📄 Download .txt", data=corrected,
                           file_name=base+"_transcript.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        srt_data = st.session_state.get("srt_data") or \
                   f"1\n00:00:00,000 --> 00:05:00,000\n{corrected}\n"
        st.download_button("🎬 Download .srt", data=srt_data,
                           file_name=base+".srt",
                           mime="text/plain", use_container_width=True)

    st.divider()
    st.subheader("Help improve the model")
    st.markdown("Did you correct any mistakes? Submit to help train a better Shona model.")

    accuracy = st.radio("How accurate was the transcription?",
                        ["Excellent","Good — minor corrections",
                         "Fair — several corrections","Poor"],
                        horizontal=True)

    if st.button("✅ Submit correction", use_container_width=True):
        with st.spinner("Saving..."):
            saved = save_correction(
                st.session_state["audio_bytes"],
                st.session_state["suffix"],
                transcript, corrected,
                st.session_state["filename"])
        if saved:
            st.success("Tatenda! Your correction has been saved as training data.")
        else:
            st.info("Contribute at: github.com/stanleymateta-tech/Project-Nyaradzai")
