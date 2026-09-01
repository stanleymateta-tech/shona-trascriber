import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Rurimi RwaAmai — Shona Language Services",
    page_icon="🎙",
    layout="centered"
)

st.title("🎙 Rurimi RwaAmai 🇿🇼")
st.markdown("""
**Mother Tongue — AI-powered Shona language services.**
Transcribe speech. Read text aloud. All in chiShona.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎤 Transcribe (Speech → Text)", "🔊 Read Aloud (Text → Speech)"])

# ── Shared helpers ────────────────────────────────────────────────────────────
SUPPORTED_TYPES = [
    "wav","flac","ogg","mp3","m4a","mp4","mov","avi","mkv","webm","aac","opus"
]

def get_audio_array(audio_bytes, suffix):
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
            if len(arr.shape) > 1: arr = arr.mean(axis=1)
            return arr.astype("float32")
        arr, sr = sf.read(tmp_in)
        if len(arr.shape) > 1: arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
        return arr.astype("float32")
    finally:
        for p in [tmp_in, tmp_wav]:
            try:
                if os.path.exists(p): os.unlink(p)
            except: pass

def save_correction(audio_bytes, suffix, original, corrected, filename):
    try:
        from huggingface_hub import HfApi
        import tempfile, os, json
        from datetime import datetime
        api = HfApi(); ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        rid = "Starsm91/shona-corrections"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes); tmp = f.name
        api.upload_file(path_or_fileobj=tmp,
                        path_in_repo=f"audio/{ts}{suffix}",
                        repo_id=rid, repo_type="dataset")
        os.unlink(tmp)
        meta = json.dumps({"timestamp":ts,"filename":filename,
                           "original":original,"corrected":corrected},
                          ensure_ascii=False)
        api.upload_file(path_or_fileobj=meta.encode(),
                        path_in_repo=f"corrections/{ts}.json",
                        repo_id=rid, repo_type="dataset")
        return True
    except: return False

# ── TAB 1: TRANSCRIPTION ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload audio or video → get Shona transcript")
    st.caption("WAV · FLAC · OGG · MP3 · M4A · MP4 · MOV · AVI · MKV · WEBM · OPUS")

    with st.sidebar:
        st.header("⚙️ Options")
        use_diarisation = st.toggle("👥 Identify speakers", value=False)
        use_noise = st.toggle("🔇 Reduce background noise", value=True)
        HF_TOKEN = st.secrets.get("HF_TOKEN", None)
        if use_diarisation and not HF_TOKEN:
            HF_TOKEN = st.text_input("Hugging Face token", type="password")
        st.divider()
        st.markdown("[Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)")
        st.markdown("[Shona ASR Model](https://huggingface.co/Starsm91/whisper-small-shona)")
        st.markdown("[Shona TTS Model](https://huggingface.co/facebook/mms-tts-sna)")

    uploaded = st.file_uploader("Choose a file", type=SUPPORTED_TYPES, key="asr_upload")

    if uploaded:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        audio_bytes = uploaded.read()
        if suffix in [".mp4",".mov",".avi",".mkv",".webm"]:
            st.video(uploaded)
        else:
            st.audio(uploaded)

        if st.button("🎙 Transcribe", type="primary", use_container_width=True):
            import transformers, numpy as np
            transformers.logging.set_verbosity_error()
            from transformers import pipeline

            with st.spinner("Extracting audio..."):
                audio_array = get_audio_array(audio_bytes, suffix)

            if use_noise:
                with st.spinner("Reducing background noise..."):
                    try:
                        import noisereduce as nr
                        audio_array = nr.reduce_noise(y=audio_array, sr=16000)
                    except: pass

            transcript = ""
            if use_diarisation and HF_TOKEN:
                with st.spinner("Identifying speakers..."):
                    try:
                        import tempfile, soundfile as sf
                        from pyannote.audio import Pipeline
                        import torch
                        tmp_wav = tempfile.mktemp(suffix=".wav")
                        sf.write(tmp_wav, audio_array, 16000)
                        pp = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-community-1", token=HF_TOKEN)
                        output = pp(tmp_wav)
                        segments = [(t.start,t.end,s)
                                    for t,_,s in output.itertracks(yield_label=True)]
                        asr = pipeline("automatic-speech-recognition",
                                       model="Starsm91/whisper-small-shona",
                                       generate_kwargs={"language":"shona","task":"transcribe"})
                        lines = []
                        for start,end,spk in segments:
                            s=int(start*16000); e=int(end*16000)
                            seg=audio_array[s:e]
                            if len(seg)<1600: continue
                            txt=asr(seg.astype(float),
                                    generate_kwargs={"language":"shona","task":"transcribe"})["text"].strip()
                            if txt:
                                lines.append(f"[{int(start//60):02d}:{int(start%60):02d}] "
                                             f"{spk.replace('SPEAKER_','Speaker ')}: {txt}")
                        transcript = "\n\n".join(lines)
                        import os; os.unlink(tmp_wav)
                    except Exception as e:
                        st.warning(f"Speaker ID failed: {e} — transcribing without labels")

            if not transcript:
                with st.spinner("Transcribing Shona... (1-2 minutes)"):
                    asr = pipeline("automatic-speech-recognition",
                                   model="Starsm91/whisper-small-shona",
                                   generate_kwargs={"language":"shona","task":"transcribe"})
                    transcript = asr(audio_array, return_timestamps=True,
                                     generate_kwargs={"language":"shona","task":"transcribe"})["text"].strip()

            st.session_state["transcript"]  = transcript
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["suffix"]      = suffix
            st.session_state["filename"]    = uploaded.name

    if st.session_state.get("transcript"):
        transcript = st.session_state["transcript"]
        st.success("Transcription complete!")
        corrected = st.text_area("Review and correct if needed:",
                                  value=transcript, height=200)
        base = st.session_state["filename"].rsplit(".",1)[0]
        col1,col2 = st.columns(2)
        with col1:
            st.download_button("📄 Download .txt", data=corrected,
                               file_name=base+"_transcript.txt",
                               mime="text/plain", use_container_width=True)
        with col2:
            srt = f"1\n00:00:00,000 --> 00:05:00,000\n{corrected}\n"
            st.download_button("🎬 Download .srt", data=srt,
                               file_name=base+".srt",
                               mime="text/plain", use_container_width=True)
        st.divider()
        st.subheader("Help improve the model")
        st.markdown("Correct any mistakes above then submit to help train a better Shona model.")
        if st.button("✅ Submit correction", use_container_width=True):
            with st.spinner("Saving..."):
                saved = save_correction(st.session_state["audio_bytes"],
                                        st.session_state["suffix"],
                                        transcript, corrected,
                                        st.session_state["filename"])
            st.success("Tatenda! Saved as training data.") if saved else \
            st.info("Contribute at github.com/stanleymateta-tech/Project-Nyaradzai")

# ── TAB 2: TEXT TO SPEECH ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Type Shona text → hear it spoken aloud")
    st.markdown("""
    Powered by Meta's MMS-TTS Shona voice model.
    Type any Shona text — a sentence, a paragraph, a novel passage — and download the audio.
    """)

    shona_text = st.text_area(
        "Enter Shona text:",
        placeholder="Mangwanani. Ndinotenda chaizvo nerubatsiro rwenyu. "
                    "Rurimi rwedu nderwedu tose...",
        height=200,
        help="Type or paste any Shona text. Longer text takes more time to process."
    )

    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speaking speed", min_value=0.5, max_value=2.0,
                          value=1.0, step=0.1,
                          help="1.0 is normal speed")
    with col2:
        st.markdown("**Voice:** Shona (Meta MMS-TTS)")
        st.markdown("**Model:** facebook/mms-tts-sna")

    if st.button("🔊 Read Aloud", type="primary",
                 use_container_width=True, disabled=not shona_text.strip()):
        with st.spinner("Generating Shona audio..."):
            try:
                import transformers, numpy as np, io
                transformers.logging.set_verbosity_error()
                from transformers import pipeline
                import soundfile as sf

                tts = pipeline("text-to-speech", model="facebook/mms-tts-sna")
                result = tts(shona_text.strip())

                audio = np.array(result["audio"]).squeeze()
                sr    = result["sampling_rate"]

                # Apply speed adjustment
                if speed != 1.0:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=speed)

                # Save to buffer
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV")
                buf.seek(0)
                audio_bytes_out = buf.read()

                st.success("Audio generated!")
                st.audio(audio_bytes_out, format="audio/wav")

                st.download_button(
                    label="⬇️ Download audio (.wav)",
                    data=audio_bytes_out,
                    file_name="shona_audio.wav",
                    mime="audio/wav",
                    use_container_width=True
                )

                # Show word count and duration info
                word_count = len(shona_text.split())
                duration   = len(audio) / sr
                st.caption(f"{word_count} words · {duration:.1f} seconds of audio")

            except Exception as e:
                st.error(f"Could not generate audio: {str(e)}")
                st.info("Try shorter text if you get a timeout error.")

    st.divider()
    st.markdown("""
    ### Use cases:
    - **Audiobooks** — paste a chapter from a Shona novel and download the audio
    - **Church** — generate audio versions of written sermons or announcements
    - **Education** — reading tools for Shona literacy
    - **Accessibility** — for visually impaired Shona speakers
    - **Content creation** — Shona voiceovers for videos

    ### About the voice:
    Powered by [Meta's MMS-TTS](https://huggingface.co/facebook/mms-tts-sna) Shona model,
    part of the Massively Multilingual Speech project covering 1,100+ languages.
    """)
