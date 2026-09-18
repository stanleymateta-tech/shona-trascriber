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
        try:
            arr, sr = sf.read(tmp_in)
            if len(arr.shape) > 1: arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
            return arr.astype("float32")
        except Exception:
            pass
        try:
            import librosa
            arr, _ = librosa.load(tmp_in, sr=16000, mono=True)
            return arr.astype("float32")
        except Exception:
            pass
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_in)
            audio = audio.set_channels(1).set_frame_rate(16000)
            arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
            arr /= np.iinfo(audio.array_type).max
            return arr
        except Exception:
            pass
        try:
            import av
            container = av.open(tmp_in)
            samples = []
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                frame.pts = None
                resampled = resampler.resample(frame)
                for r in resampled:
                    samples.append(r.to_ndarray().flatten())
            if samples:
                return np.concatenate(samples).astype(np.float32)
        except Exception:
            pass
        import subprocess, shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            result = subprocess.run([ffmpeg_path,"-y","-i",tmp_in,
                            "-ar","16000","-ac","1","-f","wav",tmp_wav],
                            capture_output=True)
            if result.returncode == 0 and os.path.exists(tmp_wav):
                arr, _ = sf.read(tmp_wav)
                if len(arr.shape) > 1: arr = arr.mean(axis=1)
                return arr.astype("float32")
        raise ValueError(f"Could not decode {suffix} file. Try WAV or FLAC format.")
    finally:
        for p in [tmp_in, tmp_wav]:
            try:
                if os.path.exists(p): os.unlink(p)
            except: pass

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_timestamped_transcript(chunks):
    """Build a readable timestamped transcript from Whisper chunks."""
    lines = []
    for chunk in chunks:
        if not chunk.get("text","").strip():
            continue
        start = chunk.get("timestamp", [0, 0])[0] or 0
        text  = chunk["text"].strip()
        lines.append(f"[{format_timestamp(start)}] {text}")
    return "\n".join(lines)

def build_srt(chunks):
    """Build proper SRT subtitle file from Whisper timestamp chunks."""
    srt_lines = []
    idx = 1
    for chunk in chunks:
        text = chunk.get("text","").strip()
        if not text:
            continue
        ts   = chunk.get("timestamp", [0, 1])
        start = ts[0] if ts[0] is not None else 0
        end   = ts[1] if ts[1] is not None else start + 3
        srt_lines.append(
            f"{idx}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{text}\n"
        )
        idx += 1
    return "\n".join(srt_lines)

def send_transcript_email(to_email, filename, transcript, srt_content):
    """Send transcript by email using SMTP."""
    try:
        import smtplib, ssl
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        # Get email credentials from Streamlit secrets
        smtp_host  = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port  = int(st.secrets.get("SMTP_PORT", 587))
        smtp_user  = st.secrets.get("SMTP_USER", "")
        smtp_pass  = st.secrets.get("SMTP_PASS", "")

        if not smtp_user or not smtp_pass:
            return False, "Email not configured. Download your transcript using the buttons above."

        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = to_email
        msg["Subject"] = f"Your Shona Transcript — {filename}"

        body = f"""Tatenda — Thank you for using Rurimi RwaAmai!

Your Shona transcript for '{filename}' is attached.

Files attached:
• {filename}_transcript.txt — Full transcript
• {filename}.srt — Subtitle file for video editors

---
Rurimi RwaAmai — Mother Tongue
AI-powered Shona language services
shona-trascriber.streamlit.app
*Mutauro wedu, panyika yose* 🇿🇼
"""
        msg.attach(MIMEText(body, "plain"))

        # Attach transcript txt
        txt_part = MIMEBase("application", "octet-stream")
        txt_part.set_payload(transcript.encode("utf-8"))
        encoders.encode_base64(txt_part)
        txt_part.add_header("Content-Disposition",
                            f"attachment; filename={filename}_transcript.txt")
        msg.attach(txt_part)

        # Attach SRT
        srt_part = MIMEBase("application", "octet-stream")
        srt_part.set_payload(srt_content.encode("utf-8"))
        encoders.encode_base64(srt_part)
        srt_part.add_header("Content-Disposition",
                            f"attachment; filename={filename}.srt")
        msg.attach(srt_part)

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())

        return True, f"Transcript sent to {to_email}"
    except Exception as e:
        return False, f"Could not send email: {str(e)}"

def translate_to_english(shona_text):
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import pipeline
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-sn-en")
        result = translator(shona_text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        return f"Translation error: {str(e)}"

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
    with st.sidebar:
        st.header("⚙️ Options")
        use_timestamps  = st.toggle("🕐 Include timestamps", value=False,
                                    help="Show [MM:SS] timestamps next to each sentence")
        use_noise       = st.toggle("🔇 Reduce background noise", value=True)
        show_translation= st.toggle("🌍 Translate to English", value=False)
        use_diarisation = st.toggle("👥 Identify speakers", value=False)
        HF_TOKEN = st.secrets.get("HF_TOKEN", None)
        if use_diarisation and not HF_TOKEN:
            HF_TOKEN = st.text_input("Hugging Face token", type="password")
        st.divider()
        st.markdown("📧 **Email delivery**")
        email_address = st.text_input("Send transcript to email",
                                       placeholder="your@email.com",
                                       help="Enter your email to receive the transcript when done")
        st.divider()
        st.markdown("[Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)")
        st.markdown("[Shona ASR Model](https://huggingface.co/Starsm91/whisper-small-shona)")

    st.subheader("Upload audio or video → get Shona transcript")
    st.caption("WAV · FLAC · OGG · MP3 · M4A · MP4 · MOV · AVI · MKV · WEBM · OPUS")

    uploaded = st.file_uploader("Choose a file", type=SUPPORTED_TYPES, key="asr_upload")

    if uploaded:
        suffix      = "." + uploaded.name.split(".")[-1].lower()
        audio_bytes = uploaded.read()
        if suffix in [".mp4",".mov",".avi",".mkv",".webm"]:
            st.video(uploaded)
        else:
            st.audio(uploaded)

        if st.button("🎙 Transcribe", type="primary", use_container_width=True):
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import pipeline

            progress = st.progress(0, text="Extracting audio...")
            with st.spinner(""):
                audio_array = get_audio_array(audio_bytes, suffix)
            progress.progress(20, text="Audio extracted...")

            if use_noise:
                progress.progress(30, text="Reducing background noise...")
                try:
                    import noisereduce as nr
                    audio_array = nr.reduce_noise(y=audio_array, sr=16000)
                except: pass

            transcript   = ""
            chunks       = []
            srt_content  = ""

            if use_diarisation and HF_TOKEN:
                progress.progress(40, text="Identifying speakers...")
                try:
                    import tempfile, soundfile as sf
                    from pyannote.audio import Pipeline as PyPipeline
                    tmp_wav = tempfile.mktemp(suffix=".wav")
                    sf.write(tmp_wav, audio_array, 16000)
                    pp = PyPipeline.from_pretrained(
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
                            ts_str = f"[{format_timestamp(start)}] " if use_timestamps else ""
                            spk_str = spk.replace("SPEAKER_","Speaker ")
                            lines.append(f"{ts_str}{spk_str}: {txt}")
                    transcript = "\n\n".join(lines)
                    import os; os.unlink(tmp_wav)
                except Exception as e:
                    st.warning(f"Speaker ID failed: {e}")

            progress.progress(50, text="Transcribing Shona speech...")
            if not transcript:
                asr = pipeline("automatic-speech-recognition",
                               model="Starsm91/whisper-small-shona",
                               generate_kwargs={"language":"shona","task":"transcribe"})
                result = asr(audio_array, return_timestamps=True,
                             generate_kwargs={"language":"shona","task":"transcribe"})
                chunks = result.get("chunks", [])

                if use_timestamps and chunks:
                    transcript = build_timestamped_transcript(chunks)
                else:
                    transcript = result["text"].strip()

            # Build SRT from chunks
            if chunks:
                srt_content = build_srt(chunks)
            else:
                srt_content = f"1\n00:00:00,000 --> 00:05:00,000\n{transcript}\n"

            progress.progress(80, text="Finishing up...")

            # Translate if requested
            english = ""
            if show_translation and transcript:
                progress.progress(85, text="Translating to English...")
                english = translate_to_english(transcript)

            # Send email if address provided
            if email_address and email_address.strip():
                progress.progress(90, text="Sending email...")
                base = uploaded.name.rsplit(".",1)[0]
                ok, msg = send_transcript_email(
                    email_address.strip(), base, transcript, srt_content)
                if ok:
                    st.success(f"📧 {msg}")
                else:
                    st.info(f"📧 {msg}")

            progress.progress(100, text="Done!")

            st.session_state["transcript"]  = transcript
            st.session_state["english"]     = english
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["suffix"]      = suffix
            st.session_state["filename"]    = uploaded.name
            st.session_state["srt_content"] = srt_content
            st.session_state["chunks"]      = chunks

    # Show results
    if st.session_state.get("transcript") or st.session_state.get("english"):
        transcript  = st.session_state.get("transcript","")
        english     = st.session_state.get("english","")
        srt_content = st.session_state.get("srt_content","")
        chunks      = st.session_state.get("chunks",[])

        if transcript:
            st.success("Transcription complete!")

            # Word and character count
            word_count = len(transcript.split())
            char_count = len(transcript)
            st.caption(f"{word_count:,} words · {char_count:,} characters")

            st.subheader("Shona Transcript")
            corrected = st.text_area("Review and correct if needed:",
                                      value=transcript, height=200)

            # Copy button
            st.code(corrected, language=None)

            if english:
                st.subheader("🌍 English Translation")
                st.text_area("English:", value=english, height=120, key="eng_display")

            if transcript and not english:
                if st.button("🌍 Translate to English", use_container_width=True):
                    with st.spinner("Translating..."):
                        eng = translate_to_english(corrected)
                    st.session_state["english"] = eng
                    st.text_area("English Translation:", value=eng, height=120)

        base = st.session_state["filename"].rsplit(".",1)[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 Download .txt",
                               data=corrected if transcript else english,
                               file_name=base+"_transcript.txt",
                               mime="text/plain", use_container_width=True)
        with col2:
            st.download_button("🎬 Download .srt",
                               data=srt_content,
                               file_name=base+".srt",
                               mime="text/plain", use_container_width=True)
        with col3:
            if chunks:
                ts_transcript = build_timestamped_transcript(chunks)
                st.download_button("🕐 Download with timestamps",
                                   data=ts_transcript,
                                   file_name=base+"_timestamped.txt",
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
