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

Upload a Shona audio file and get a transcript instantly.
Built on [whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona) — WER 36.42%.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a Shona audio file",
    type=["mp3", "wav", "m4a", "mp4", "ogg", "flac"],
    help="Supported formats: MP3, WAV, M4A, MP4, OGG, FLAC"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Transcribe", type="primary", use_container_width=True):
        with st.spinner("Transcribing your Shona audio..."):
            try:
                import transformers
                transformers.logging.set_verbosity_error()
                from transformers import pipeline
                import tempfile, os

                # Save uploaded file to temp location
                suffix = "." + uploaded_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                # Load model and transcribe
                asr = pipeline(
                    "automatic-speech-recognition",
                    model="Starsm91/whisper-small-shona",
                    generate_kwargs={"language": "shona", "task": "transcribe"},
                )
                result = asr(
                    tmp_path,
                    return_timestamps=True,
                    generate_kwargs={"language": "shona", "task": "transcribe"},
                )
                os.unlink(tmp_path)
                text = result["text"].strip()

                if text:
                    st.success("Transcription complete!")
                    st.text_area(
                        "Shona Transcript",
                        value=text,
                        height=200,
                    )
                    st.download_button(
                        label="Download transcript as .txt",
                        data=text,
                        file_name="shona_transcript.txt",
                        mime="text/plain",
                    )
                else:
                    st.warning("No speech detected. Try a clearer recording.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.markdown("""
### Tips for best results:
- Speak clearly at normal pace
- Reduce background noise
- Works best with clear Shona speech

### About:
- GitHub: [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)
- Model: [Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona)
- Community: [Masakhane](https://masakhane.io)
""")
