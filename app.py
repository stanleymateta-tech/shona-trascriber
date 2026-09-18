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

# ── Audio extraction ──────────────────────────────────────────────────────────
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
        except Exception: pass
        try:
            import librosa
            arr, _ = librosa.load(tmp_in, sr=16000, mono=True)
            return arr.astype("float32")
        except Exception: pass
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_in)
            audio = audio.set_channels(1).set_frame_rate(16000)
            arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
            arr /= np.iinfo(audio.array_type).max
            return arr
        except Exception: pass
        try:
            import av
            container = av.open(tmp_in)
            samples = []
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                frame.pts = None
                resampled = resampler.resample(frame)
                for r in resampled: samples.append(r.to_ndarray().flatten())
            if samples: return np.concatenate(samples).astype(np.float32)
        except Exception: pass
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

# ── Timestamp helpers ─────────────────────────────────────────────────────────
def fmt_ts(s):
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

def fmt_srt(s):
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s%1)*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def build_timestamped(chunks):
    lines = []
    for c in chunks:
        if not c.get("text","").strip(): continue
        start = (c.get("timestamp") or [0])[0] or 0
        lines.append(f"[{fmt_ts(start)}] {c['text'].strip()}")
    return "\n".join(lines)

def build_srt(chunks):
    out=[]; idx=1
    for c in chunks:
        text=c.get("text","").strip()
        if not text: continue
        ts=c.get("timestamp",[0,3]) or [0,3]
        s=ts[0] or 0; e=ts[1] or s+3
        out.append(f"{idx}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{text}\n"); idx+=1
    return "\n".join(out)

# ── Spell check ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_shona_dictionary():
    """Load Shona dictionary from GitHub for spell checking."""
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/stanleymateta-tech/Project-Nyaradzai/main/dictionaries/sn_ZW.dic"
        with urllib.request.urlopen(url, timeout=10) as r:
            lines = r.read().decode("utf-8").splitlines()
        words = set()
        for i, line in enumerate(lines):
            if i == 0 or line.startswith("#") or not line.strip(): continue
            words.add(line.split("/")[0].strip().lower())
        return words
    except Exception:
        return set()

def spell_check_shona(text, dictionary):
    """Return list of (word, position) tuples for words not in dictionary."""
    import re
    if not dictionary: return []
    issues = []
    for m in re.finditer(r'\b[a-zA-Z]{2,}\b', text):
        word = m.group(0)
        if word.lower() not in dictionary:
            issues.append(word)
    return list(set(issues))

# ── Summary ───────────────────────────────────────────────────────────────────
def summarise_with_claude(transcript, english_translation):
    """Generate a summary using Claude API."""
    try:
        import urllib.request, json
        text_to_summarise = english_translation if english_translation else transcript
        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": f"Summarise this Shona speech transcript in 2-3 sentences in English. Focus on the main themes and key points.\n\nTranscript:\n{text_to_summarise[:3000]}"
            }]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type":"application/json",
                     "anthropic-version":"2023-06-01"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return data["content"][0]["text"]
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

# ── Email ─────────────────────────────────────────────────────────────────────
def send_email(to_email, filename, transcript, srt_content, summary=""):
    try:
        import smtplib, ssl
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        smtp_host = st.secrets.get("SMTP_HOST","smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT",587))
        smtp_user = st.secrets.get("SMTP_USER","")
        smtp_pass = st.secrets.get("SMTP_PASS","")

        if not smtp_user or not smtp_pass:
            return False, "Email not configured — download using the buttons below."

        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"]   = to_email
        msg["Subject"] = f"Your Shona Transcript — {filename}"

        body = f"""Tatenda — Thank you for using Rurimi RwaAmai!

Your Shona transcript for '{filename}' is attached.
"""
        if summary:
            body += f"\n📋 Summary:\n{summary}\n"
        body += "\nFiles attached:\n• Transcript (.txt)\n• Subtitles (.srt)\n\n---\nRurimi RwaAmai | shona-trascriber.streamlit.app\n*Mutauro wedu, panyika yose* 🇿🇼"
        msg.attach(MIMEText(body,"plain"))

        for content, fname in [(transcript, f"{filename}_transcript.txt"),
                                (srt_content, f"{filename}.srt")]:
            part = MIMEBase("application","octet-stream")
            part.set_payload(content.encode("utf-8"))
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={fname}")
            msg.attach(part)

        ctx = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls(context=ctx)
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, to_email, msg.as_string())
        return True, f"Transcript sent to {to_email} ✅"
    except Exception as e:
        return False, f"Email failed: {e}"

# ── Translation ───────────────────────────────────────────────────────────────
def translate_to_english(shona_text):
    try:
        import transformers; transformers.logging.set_verbosity_error()
        from transformers import pipeline
        t = pipeline("translation", model="Helsinki-NLP/opus-mt-sn-en")
        return t(shona_text, max_length=512)[0]["translation_text"]
    except Exception as e:
        return f"Translation error: {e}"

# ── Save correction ───────────────────────────────────────────────────────────
def save_correction(audio_bytes, suffix, original, corrected, filename):
    try:
        from huggingface_hub import HfApi
        import tempfile, os, json
        from datetime import datetime
        api=HfApi(); ts=datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        rid="Starsm91/shona-corrections"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
            f.write(audio_bytes); tmp=f.name
        api.upload_file(path_or_fileobj=tmp,path_in_repo=f"audio/{ts}{suffix}",
                        repo_id=rid,repo_type="dataset"); os.unlink(tmp)
        api.upload_file(path_or_fileobj=json.dumps(
            {"timestamp":ts,"filename":filename,"original":original,"corrected":corrected},
            ensure_ascii=False).encode(),
            path_in_repo=f"corrections/{ts}.json",repo_id=rid,repo_type="dataset")
        return True
    except: return False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with tab1:
    with st.sidebar:
        st.header("⚙️ Options")
        use_timestamps   = st.toggle("🕐 Timestamps", value=False)
        use_noise        = st.toggle("🔇 Reduce noise", value=True)
        show_translation = st.toggle("🌍 Translate to English", value=False)
        use_summary      = st.toggle("📋 AI Summary", value=False,
                                     help="Generate a 2-3 sentence English summary using AI")
        use_spellcheck   = st.toggle("✏️ Spell check", value=False,
                                     help="Highlight words not in the Shona dictionary")
        use_diarisation  = st.toggle("👥 Identify speakers", value=False)
        HF_TOKEN = st.secrets.get("HF_TOKEN", None)
        if use_diarisation and not HF_TOKEN:
            HF_TOKEN = st.text_input("HF token", type="password")
        st.divider()
        st.markdown("**📧 Email delivery**")
        email_address = st.text_input("Send to email",
                                       placeholder="your@email.com")
        st.divider()
        st.markdown("[Project Nyaradzai](https://github.com/stanleymateta-tech/Project-Nyaradzai)")

    # ── BATCH UPLOAD ──────────────────────────────────────────────────────────
    st.subheader("Upload audio or video → get Shona transcript")
    st.caption("WAV · FLAC · OGG · MP3 · M4A · MP4 · MOV · AVI · MKV · WEBM · OPUS")

    batch_mode = st.checkbox("📁 Batch mode — transcribe multiple files at once",
                              value=False)

    if batch_mode:
        uploaded_files = st.file_uploader(
            "Upload multiple files", type=SUPPORTED_TYPES,
            accept_multiple_files=True, key="batch_upload")

        if uploaded_files and st.button("🎙 Transcribe All", type="primary",
                                         use_container_width=True):
            import transformers, zipfile, io as io_mod
            transformers.logging.set_verbosity_error()
            from transformers import pipeline

            asr = pipeline("automatic-speech-recognition",
                           model="Starsm91/whisper-small-shona",
                           generate_kwargs={"language":"shona","task":"transcribe"})

            zip_buf = io_mod.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, uf in enumerate(uploaded_files):
                    suffix = "." + uf.name.split(".")[-1].lower()
                    audio_bytes = uf.read()
                    prog = st.progress(0, text=f"Processing {uf.name}...")
                    try:
                        arr = get_audio_array(audio_bytes, suffix)
                        prog.progress(50, text=f"Transcribing {uf.name}...")
                        result = asr(arr, return_timestamps=True,
                                     generate_kwargs={"language":"shona","task":"transcribe"})
                        chunks = result.get("chunks",[])
                        text = build_timestamped(chunks) if use_timestamps else result["text"].strip()
                        srt  = build_srt(chunks)
                        base = uf.name.rsplit(".",1)[0]
                        zf.writestr(f"{base}_transcript.txt", text)
                        zf.writestr(f"{base}.srt", srt)
                        prog.progress(100, text=f"✅ {uf.name} done")
                        st.success(f"✅ {uf.name} — {len(text.split())} words")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

            zip_buf.seek(0)
            st.download_button(
                "⬇️ Download all transcripts (.zip)",
                data=zip_buf.read(),
                file_name="shona_transcripts.zip",
                mime="application/zip",
                use_container_width=True
            )

    else:
        # ── SINGLE FILE ───────────────────────────────────────────────────────
        uploaded = st.file_uploader("Choose a file", type=SUPPORTED_TYPES,
                                     key="asr_upload")

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

                prog = st.progress(0, text="Extracting audio...")
                audio_array = get_audio_array(audio_bytes, suffix)
                prog.progress(20, text="Audio ready...")

                if use_noise:
                    prog.progress(30, text="Reducing noise...")
                    try:
                        import noisereduce as nr
                        audio_array = nr.reduce_noise(y=audio_array, sr=16000)
                    except: pass

                transcript = ""; chunks = []; srt_content = ""

                if use_diarisation and HF_TOKEN:
                    prog.progress(40, text="Identifying speakers...")
                    try:
                        import tempfile, soundfile as sf
                        from pyannote.audio import Pipeline as PP
                        tmp_wav = tempfile.mktemp(suffix=".wav")
                        sf.write(tmp_wav, audio_array, 16000)
                        pp = PP.from_pretrained("pyannote/speaker-diarization-community-1",
                                                 token=HF_TOKEN)
                        output = pp(tmp_wav)
                        segs = [(t.start,t.end,s)
                                for t,_,s in output.itertracks(yield_label=True)]
                        asr = pipeline("automatic-speech-recognition",
                                       model="Starsm91/whisper-small-shona",
                                       generate_kwargs={"language":"shona","task":"transcribe"})
                        lines=[]
                        for start,end,spk in segs:
                            seg=audio_array[int(start*16000):int(end*16000)]
                            if len(seg)<1600: continue
                            txt=asr(seg.astype(float),
                                    generate_kwargs={"language":"shona","task":"transcribe"})["text"].strip()
                            if txt:
                                ts_str = f"[{fmt_ts(start)}] " if use_timestamps else ""
                                lines.append(f"{ts_str}{spk.replace('SPEAKER_','Speaker ')}: {txt}")
                        transcript="\n\n".join(lines)
                        import os; os.unlink(tmp_wav)
                    except Exception as e:
                        st.warning(f"Speaker ID failed: {e}")

                if not transcript:
                    prog.progress(50, text="Transcribing Shona...")
                    asr = pipeline("automatic-speech-recognition",
                                   model="Starsm91/whisper-small-shona",
                                   generate_kwargs={"language":"shona","task":"transcribe"})
                    result = asr(audio_array, return_timestamps=True,
                                 generate_kwargs={"language":"shona","task":"transcribe"})
                    chunks = result.get("chunks",[])
                    transcript = build_timestamped(chunks) if (use_timestamps and chunks) \
                                 else result["text"].strip()

                srt_content = build_srt(chunks) if chunks else \
                              f"1\n00:00:00,000 --> 00:05:00,000\n{transcript}\n"

                english = ""
                if show_translation:
                    prog.progress(70, text="Translating to English...")
                    english = translate_to_english(transcript)

                summary = ""
                if use_summary:
                    prog.progress(80, text="Generating AI summary...")
                    summary = summarise_with_claude(transcript, english)

                # Spell check
                spell_issues = []
                if use_spellcheck:
                    prog.progress(85, text="Spell checking...")
                    dictionary = load_shona_dictionary()
                    spell_issues = spell_check_shona(transcript, dictionary)

                # Email
                if email_address and email_address.strip():
                    prog.progress(90, text="Sending email...")
                    ok, msg = send_email(email_address.strip(),
                                         uploaded.name.rsplit(".",1)[0],
                                         transcript, srt_content, summary)
                    st.success(msg) if ok else st.info(msg)

                prog.progress(100, text="Done!")

                st.session_state.update({
                    "transcript": transcript, "english": english,
                    "summary": summary, "spell_issues": spell_issues,
                    "audio_bytes": audio_bytes, "suffix": suffix,
                    "filename": uploaded.name, "srt_content": srt_content,
                    "chunks": chunks
                })

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.get("transcript"):
        transcript   = st.session_state["transcript"]
        english      = st.session_state.get("english","")
        summary      = st.session_state.get("summary","")
        spell_issues = st.session_state.get("spell_issues",[])
        srt_content  = st.session_state.get("srt_content","")
        chunks       = st.session_state.get("chunks",[])

        st.success("Transcription complete!")
        word_count = len(transcript.split())
        char_count = len(transcript)
        st.caption(f"**{word_count:,} words · {char_count:,} characters**")

        # Summary box
        if summary:
            st.info(f"📋 **Summary:** {summary}")

        st.subheader("Shona Transcript")
        corrected = st.text_area("Review and correct if needed:",
                                  value=transcript, height=200)

        # Spell check results
        if spell_issues:
            with st.expander(f"✏️ {len(spell_issues)} possible spelling issues"):
                st.markdown("These words were not found in the Shona dictionary. "
                            "They may be correct — proper names and rare words may not be listed.")
                st.write(", ".join(sorted(spell_issues)))

        # Copy hint
        st.caption("💡 Click the text area above and press Ctrl+A then Ctrl+C to copy all")

        if english:
            st.subheader("🌍 English Translation")
            st.text_area("English:", value=english, height=120, key="eng_out")

        if transcript and not english and show_translation:
            if st.button("🌍 Translate to English", use_container_width=True):
                with st.spinner("Translating..."):
                    eng = translate_to_english(corrected)
                st.session_state["english"] = eng
                st.text_area("English:", value=eng, height=120)

        # Download buttons
        base = st.session_state["filename"].rsplit(".",1)[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 .txt", data=corrected,
                               file_name=base+"_transcript.txt",
                               mime="text/plain", use_container_width=True)
        with col2:
            st.download_button("🎬 .srt", data=srt_content,
                               file_name=base+".srt",
                               mime="text/plain", use_container_width=True)
        with col3:
            if chunks:
                st.download_button("🕐 Timestamped",
                                   data=build_timestamped(chunks),
                                   file_name=base+"_timestamped.txt",
                                   mime="text/plain", use_container_width=True)

        st.divider()
        st.subheader("Help improve the model")
        if st.button("✅ Submit correction", use_container_width=True):
            with st.spinner("Saving..."):
                saved = save_correction(st.session_state["audio_bytes"],
                                        st.session_state["suffix"],
                                        transcript, corrected,
                                        st.session_state["filename"])
            st.success("Tatenda! Saved as training data.") if saved else \
            st.info("Contribute at github.com/stanleymateta-tech/Project-Nyaradzai")

# ── TAB 2: TTS ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Type Shona text → hear it spoken aloud")
    st.markdown("Powered by Meta MMS-TTS Shona voice.")

    shona_text = st.text_area("Enter Shona text:",
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
                sf.write(buf, audio, sr, format="WAV"); buf.seek(0)
                audio_out = buf.read()
                st.success("Audio generated!")
                st.audio(audio_out, format="audio/wav")
                st.download_button("⬇️ Download .wav", data=audio_out,
                                   file_name="shona_audio.wav", mime="audio/wav",
                                   use_container_width=True)
                st.caption(f"{len(shona_text.split())} words · {len(audio)/sr:.1f} seconds")
            except Exception as e:
                st.error(f"Could not generate audio: {e}")

    st.divider()
    st.markdown("""
    ### Use cases:
    - **Audiobooks** — paste Shona novel text and download audio
    - **Church** — generate audio versions of written sermons
    - **Education** — reading tools for Shona literacy
    - **Accessibility** — for visually impaired Shona speakers
    """)
