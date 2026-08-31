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

AWS_SERVER = None  # Set to "http://YOUR-AWS-IP:5000" when running

SUPPORTED_TYPES = [
    "wav", "flac", "ogg", "mp3", "m4a", "mp4",
    "mov", "avi", "mkv", "webm", "aac", "opus"
]

def extract_audio_ffmpeg(input_path, output_path):
    """Use ffmpeg binary to convert any format to 16kHz WAV."""
    import subprocess
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path
    ], capture_output=True, text=True)
    return result.returncode == 0

def get_audio_array(audio_bytes, suffix):
    """Convert uploaded file to numpy float32 array at 16kHz."""
    import numpy as np
    import tempfile, os
    import soundfile as sf

    # Write uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_in = f.name

    tmp_wav = tmp_in.replace(suffix, "_converted.wav")

    try:
        # Try direct soundfile read first (WAV, FLAC, OGG)
        try:
            audio_array, sr = sf.read(tmp_in)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(
                    audio_array.astype(np.float32),
                    orig_sr=sr, target_sr=16000)
            return audio_array.astype(np.float32)
        except Exception:
            pass

        # Use ffmpeg to convert to WAV then read
        success = extract_audio_ffmpeg(tmp_in, tmp_wav)
        if success and os.path.exists(tmp_wav):
            audio_array, sr = sf.read(tmp_wav)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32)

        # Last resort: librosa
        import librosa
        audio_array, _ = librosa.load(tmp_in, sr=16000, mono=True)
        return audio_array.astype(np.float32)

    finally:
        for p in [tmp_in, tmp_wav]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

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
st.caption("Supported: WAV · FLAC · OGG · MP3 · M4A · AAC · MP4 · MOV · AVI · MKV · WEBM · OPUS")

uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=SUPPORTED_TYPES,
    help="Any audio or video file containing Shona speech"
)

if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    audio_bytes = uploaded_file.read()

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
                with st.spinner("Processing... this takes 1-2 minutes"):
                    audio_array = get_audio_array(audio_bytes, suffix)
                    text = transcribe_local(audio_array)

            if text:
                st.success("Transcription complete!")
                st.text_area("Shona Transcript", value=text, height=200)
                col1, col2 = st.columns(2)
                base_name = uploaded_file.name.rsplit(".", 1)[0]
                with col1:
                    st.download_button(
                        label="📄 Download as .txt",
                        data=text,
                        file_name=base_name + "_transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col2:
                    srt = f"1\n00:00:00,000 --> 00:05:00,000\n{text}\n"
                    st.download_button(
                        label="🎬 Download as .srt (subtitles)",
                        data=srt,
                        file_name=base_name + ".srt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            else:
                st.warning("No speech detected. Try a clearer recording.")

        except Exception as e:
            st.error(f"Could not process file: {str(e)}")
            st.info("Try converting your file to WAV using a free online converter, then upload again.")

st.divider()
st.markdown("""
### Tips for best results:
- Any audio or video format is supported
- Speak clearly at normal pace
- Reduce background noise where possible
- Longer files take more time on CPU

### About:
- GitHub: [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)
- Model: [Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona)
- Community: [Masakhane](https://masakhane.io)
""")
