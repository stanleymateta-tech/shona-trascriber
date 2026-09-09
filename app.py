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
Transcribe · Translate · Read Aloud. All in chiShona.

Part of [Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai) —
*Mutauro wedu, panyika yose* 🇿🇼
""")

st.divider()

tab1, tab2 = st.tabs(["🎤 Transcribe (Speech → Text)", "🔊 Read Aloud (Text → Speech)"])

SUPPORTED_TYPES = [
    "wav","flac","ogg","mp3","m4a","mp4","mov","avi","mkv","webm","aac","opus"
]

def get_audio_array(audio_bytes, suffix):
    import numpy as np, tempfile, os, soundfile as sf
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes); tmp_in = f.name
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        import subprocess, shutil
        ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        result = subprocess.run([ffmpeg_path,"-y","-i",tmp_in,"-ar","16000","-ac","1",
                        "-f","wav",tmp_wav], capture_output=True)
        if result.returncode == 0 and os.path.exists(tmp_wav):
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

def translate_to_english(shona_text):
    """Translate Shona text to English using Helsinki-NLP opus-mt-sn-en."""
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import pipeline
        translator = pipeline("translation",
                              model="Helsinki-NLP/opus-mt-sn-en")
        result = translator(shona_text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        # Fallback: use Whisper translation mode
        return f"Translation error: {str(e)}"

def whisper_translate(audio_array):
    """Direct Shona speech to English using Whisper translation mode."""
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="Starsm91/whisper-small-shona",
                   generate_kwargs={"language": "shona", "task": "translate"})
    result = asr(audio_array, return_timestamps=True,
                 generate_kwargs={"language": "shona", "task": "translate"})
    return result["text"].strip()

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

# ── TAB 1: TRANSCRIPTION + TRANSLATION ───────────────────────────────────────
with tab1:
    with st.sidebar:
        st.header("⚙️ Options")
        use_diarisation = st.toggle("👥 Identify speakers", value=False)
        use_noise = st.toggle("🔇 Reduce background noise", value=True)
        show_translation = st.toggle("🌍 Translate to English", value=False)
        direct_translate = st.toggle("⚡ Direct speech→English (faster)",
                                     value=False,
                                     help="Translates directly from audio without Shona text step")
        HF_TOKEN = st.secrets.get("HF_TOKEN", None)
        if use_diarisation and not HF_TOKEN:
            HF_TOKEN = st.text_input("Hugging Face token", type="password")
        st.divider()
        st.markdown("[Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)")
        st.markdown("[Shona ASR Model](https://huggingface.co/Starsm91/whisper-small-shona)")

    st.subheader("Upload audio or video → get Shona transcript")
    st.caption("WAV · FLAC · OGG · MP3 · M4A · MP4 · MOV · AVI · MKV · WEBM · OPUS")

    uploaded = st.file_uploader("Choose a file", type=SUPPORTED_TYPES, key="asr_upload")

    if uploaded:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        audio_bytes = uploaded.read()
        if suffix in [".mp4",".mov",".avi",".mkv",".webm"]:
            st.video(uploaded)
        else:
            st.audio(uploaded)

        if st.button("🎙 Transcribe", type="primary", use_container_width=True):
            import transformers
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

            # Direct speech-to-English translation
            if direct_translate:
                with st.spinner("Translating Shona speech to English directly..."):
                    english_text = whisper_translate(audio_array)
                st.session_state["transcript"] = ""
                st.session_state["english"] = english_text
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["suffix"] = suffix
                st.session_state["filename"] = uploaded.name

            else:
                # Normal transcription
                transcript = ""
                if use_diarisation and HF_TOKEN:
                    with st.spinner("Identifying speakers..."):
                        try:
                            import tempfile, soundfile as sf
                            from pyannote.audio import Pipeline
                            tmp_wav = tempfile.mktemp(suffix=".wav")
                            sf.write(tmp_wav, audio_array, 16000)
                            pp = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-community-1",
                                token=HF_TOKEN)
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
                                        generate_kwargs={"language":"shona",
                                                         "task":"transcribe"})["text"].strip()
                                if txt:
                                    lines.append(
                                        f"[{int(start//60):02d}:{int(start%60):02d}] "
                                        f"{spk.replace('SPEAKER_','Speaker ')}: {txt}")
                            transcript = "\n\n".join(lines)
                            import os; os.unlink(tmp_wav)
                        except Exception as e:
                            st.warning(f"Speaker ID failed: {e}")

                if not transcript:
                    with st.spinner("Transcribing Shona... (1-2 minutes)"):
                        asr = pipeline("automatic-speech-recognition",
                                       model="Starsm91/whisper-small-shona",
                                       generate_kwargs={"language":"shona","task":"transcribe"})
                        transcript = asr(audio_array, return_timestamps=True,
                                         generate_kwargs={"language":"shona",
                                                          "task":"transcribe"})["text"].strip()

                # Translate if requested
                english = ""
                if show_translation and transcript:
                    with st.spinner("Translating to English..."):
                        english = translate_to_english(transcript)

                st.session_state["transcript"]  = transcript
                st.session_state["english"]     = english
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["suffix"]      = suffix
                st.session_state["filename"]    = uploaded.name

    # Show results
    if st.session_state.get("transcript") or st.session_state.get("english"):
        transcript = st.session_state.get("transcript","")
        english    = st.session_state.get("english","")

        if transcript:
            st.success("Transcription complete!")
            st.subheader("Shona Transcript")
            corrected = st.text_area("Review and correct if needed:",
                                      value=transcript, height=180)

            # Show English translation
            if english:
                st.subheader("🌍 English Translation")
                st.text_area("English:", value=english, height=120,
                             key="eng_display")

            # If translation not yet done, offer button
            if transcript and not english:
                if st.button("🌍 Translate to English", use_container_width=True):
                    with st.spinner("Translating..."):
                        eng = translate_to_english(corrected)
                    st.session_state["english"] = eng
                    st.text_area("English Translation:", value=eng, height=120)

        elif english:
            st.success("Translation complete!")
            st.subheader("🌍 English Translation (direct from speech)")
            st.text_area("English:", value=english, height=200)
            corrected = english

        base = st.session_state["filename"].rsplit(".",1)[0]
        col1,col2 = st.columns(2)
        with col1:
            download_text = corrected if transcript else english
            st.download_button("📄 Download .txt", data=download_text,
                               file_name=base+"_transcript.txt",
                               mime="text/plain", use_container_width=True)
        with col2:
            srt_text = corrected if transcript else english
            srt = f"1\n00:00:00,000 --> 00:05:00,000\n{srt_text}\n"
            st.download_button("🎬 Download .srt", data=srt,
                               file_name=base+".srt",
                               mime="text/plain", use_container_width=True)

        if transcript:
            st.divider()
            st.subheader("Help improve the model")
            st.markdown("Correct any mistakes above then submit as training data.")
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
    st.markdown("Powered by Meta MMS-TTS Shona voice.")

    shona_text = st.text_area(
        "Enter Shona text:",
        placeholder="Mangwanani. Ndinotenda chaizvo nerubatsiro rwenyu...",
        height=180)

    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speaking speed", 0.5, 2.0, 1.0, 0.1)
    with col2:
        st.markdown("**Voice:** Shona (Meta MMS-TTS)")

    if st.button("🔊 Read Aloud", type="primary",
                 use_container_width=True, disabled=not shona_text.strip()):
        with st.spinner("Generating Shona audio..."):
            try:
                import transformers, numpy as np, io, soundfile as sf
                transformers.logging.set_verbosity_error()
                from transformers import pipeline

                tts = pipeline("text-to-speech", model="facebook/mms-tts-sna")
                result = tts(shona_text.strip())
                audio = np.array(result["audio"]).squeeze()
                sr    = result["sampling_rate"]

                if speed != 1.0:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=speed)

                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV")
                buf.seek(0)
                audio_bytes_out = buf.read()

                st.success("Audio generated!")
                st.audio(audio_bytes_out, format="audio/wav")
                st.download_button("⬇️ Download audio (.wav)",
                                   data=audio_bytes_out,
                                   file_name="shona_audio.wav",
                                   mime="audio/wav",
                                   use_container_width=True)

                duration = len(audio)/sr
                st.caption(f"{len(shona_text.split())} words · {duration:.1f} seconds")

            except Exception as e:
                st.error(f"Could not generate audio: {str(e)}")

    st.divider()
    st.markdown("""
    ### Use cases:
    - **Audiobooks** — paste Shona novel text and download audio
    - **Church** — generate audio versions of written sermons
    - **Education** — reading tools for Shona literacy
    - **Accessibility** — for visually impaired Shona speakers
    """)
