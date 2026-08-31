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

Upload any audio or video file and get a Shona transcript instantly.
Built on [whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona) — WER 36.42%.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

AWS_SERVER = None  # Set to "http://YOUR-AWS-IP:5000" when server is running

SUPPORTED_TYPES = [
    "wav", "flac", "ogg", "mp3", "m4a", "mp4",
    "mov", "avi", "mkv", "webm", "aac", "wma", "opus"
]

def extract_audio(audio_bytes, suffix):
    """Extract and convert audio to 16kHz mono float32 numpy array."""
    import numpy as np
    import io
    import soundfile as sf

    # Try soundfile first (WAV, FLAC, OGG, OPUS work natively)
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
        return audio_array.astype(np.float32)
    except Exception:
        pass

    # Fall back to pydub which handles MP3, M4A, AAC, WMA, video files
    try:
        from pydub import AudioSegment
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_in = f.name
        audio = AudioSegment.from_file(tmp_in)
        audio = audio.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(audio.array_type).max
        os.unlink(tmp_in)
        return samples
    except Exception:
        pass

    # Last resort — write to disk and let transformers/librosa handle it
    import tempfile, os
    import librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_in = f.name
    audio_array, sr = librosa.load(tmp_in, sr=16000, mono=True)
    os.unlink(tmp_in)
    return audio_array.astype(np.float32)

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

def transcribe_aws(audio_bytes, server_url, suffix):
    import requests, io
    response = requests.post(
        f"{server_url}/transcribe",
        files={"audio": (f"audio{suffix}", io.BytesIO(audio_bytes))},
        timeout=60
    )
    data = response.json()
    if data.get("success"):
        return data["text"]
    raise Exception(data.get("error", "Server error"))

def check_aws(server_url):
    import requests
    try:
        r = requests.get(f"{server_url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# Connection status
if AWS_SERVER:
    if check_aws(AWS_SERVER):
        st.success("⚡ Connected to AWS GPU server — fast transcription active")
    else:
        st.warning("⚠️ AWS server not reachable — using local CPU (slower)")
        AWS_SERVER = None
else:
    st.info("ℹ️ Using local CPU — transcription takes 1-2 minutes per file")

st.subheader("Upload a file")
st.caption("Supported: WAV · FLAC · OGG · MP3 · M4A · AAC · WMA · MP4 · MOV · AVI · MKV · WEBM · OPUS")

uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=SUPPORTED_TYPES,
    help="Any audio or video file containing Shona speech"
)

if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    audio_bytes = uploaded_file.read()

    # Show preview if audio/video
    if suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        st.video(uploaded_file)
        st.caption("Video uploaded — will extract and transcribe the audio track")
    else:
        st.audio(uploaded_file)

    if st.button("🎙 Transcribe", type="primary", use_container_width=True):
        try:
            if AWS_SERVER:
                with st.spinner("Transcribing on AWS GPU server..."):
                    text = transcribe_aws(audio_bytes, AWS_SERVER, suffix)
            else:
                with st.spinner("Extracting audio and transcribing... (1-2 minutes)"):
                    audio_array = extract_audio(audio_bytes, suffix)
                    text = transcribe_local(audio_array)

            if text:
                st.success("Transcription complete!")
                st.text_area("Shona Transcript", value=text, height=200)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📄 Download as .txt",
                        data=text,
                        file_name=uploaded_file.name.rsplit(".", 1)[0] + "_transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col2:
                    # Also offer SRT subtitle format
                    srt = f"1\n00:00:00,000 --> 00:05:00,000\n{text}\n"
                    st.download_button(
                        label="🎬 Download as .srt (subtitles)",
                        data=srt,
                        file_name=uploaded_file.name.rsplit(".", 1)[0] + ".srt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            else:
                st.warning("No speech detected. Please try a clearer recording.")

        except Exception as e:
            st.error(f"Could not process file: {str(e)}")
            st.info("If your file is not working, try converting it to WAV first "
                    "using any free online converter, then upload again.")

st.divider()
st.markdown("""
### Tips for best results:
- Any audio or video format works
- Speak clearly at normal pace  
- Reduce background noise where possible
- Longer files take more time on CPU

### About:
- GitHub: [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)
- Model: [Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona)
- Community: [Masakhane](https://masakhane.io)
""")
