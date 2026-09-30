import asyncio, json, os
from pathlib import Path
import pipeline

W, H = pipeline.W, pipeline.H


async def download(url: str, d: Path) -> Path:
    p = await asyncio.create_subprocess_exec(
        "yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
        "--no-playlist", "-o", "source.%(ext)s", url, cwd=d, stderr=asyncio.subprocess.PIPE)
    _, err = await p.communicate()
    if p.returncode:
        raise RuntimeError("Download failed: " + err.decode()[-300:])
    return next(d.glob("source.*"))


def transcribe(path: Path):
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), compute_type="int8")
    it, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    segs, words = [], []
    for s in it:
        segs.append((s.start, s.text.strip()))
        words += [(w.word.strip(), w.start, w.end) for w in (s.words or [])]
    return segs, words, info.duration


def pick_clips(segs, n, duration):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY so Claude can choose the clips.")
    import anthropic
    text = "\n".join(f"[{int(s)}] {t}" for s, t in segs)
    r = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5", max_tokens=1500,
        messages=[{"role": "user", "content":
            f"Below is a timestamped transcript ([seconds] text). Choose the {n} best moments "
            "for standalone short-form clips. Each must be 20-60 seconds, open on a strong hook, "
            "end on a complete thought, and not overlap. Reply with ONLY a JSON array of objects "
            f"with keys start, end (seconds), title (under 8 words).\n\n{text}"}])
    raw = r.content[0].text
    clips = []
    for c in json.loads(raw[raw.index("["): raw.rindex("]") + 1]):
        s, e = float(c["start"]), float(c["end"])
        if 10 <= e - s <= 90 and 0 <= s < e <= duration + 1:
            clips.append({"start": s, "end": e, "title": str(c.get("title", "Clip"))})
    if not clips:
        raise RuntimeError("Claude returned no usable clips. Try again.")
    return clips


async def render_clip(src: Path, clip: dict, words, d: Path, name: str):
    cw = [(w, max(0, s - clip["start"]), e - clip["start"]) for w, s, e in words
          if s >= clip["start"] - 0.05 and e <= clip["end"] + 0.05]
    pipeline.write_ass(cw, d / f"{name}.ass")
    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},ass={name}.ass"
    cmd = ["ffmpeg", "-y", "-ss", f"{clip['start']:.2f}", "-t", f"{clip['end'] - clip['start']:.2f}",
           "-i", str(src.resolve()), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", f"{name}.mp4"]
    p = await asyncio.create_subprocess_exec(*cmd, cwd=d, stderr=asyncio.subprocess.PIPE)
    _, err = await p.communicate()
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + err.decode()[-400:])
