import asyncio, base64, os, random, secrets, uuid
from pathlib import Path
from fastapi import FastAPI, File, Response, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import clipper, pipeline

app = FastAPI()


@app.middleware("http")
async def password_gate(request, call_next):
    pw = os.getenv("APP_PASSWORD")
    if pw:
        ok, hdr = False, request.headers.get("authorization", "")
        if hdr.startswith("Basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(hdr[6:]).decode().partition(":")[2], pw)
            except Exception:
                pass
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Clipforge"'})
    return await call_next(request)
OUT, BG = Path("outputs"), Path("backgrounds")
OUT.mkdir(exist_ok=True); BG.mkdir(exist_ok=True)
jobs: dict[str, dict] = {}


class Req(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    use_ai_script: bool = True
    voice: str = "en-US-GuyNeural"
    background: str = ""


async def run(job_id: str, r: Req):
    try:
        jobs[job_id] = {"status": "Writing script"}
        script = await asyncio.to_thread(pipeline.write_script, r.prompt) if r.use_ai_script else r.prompt
        jobs[job_id] = {"status": "Rendering video", "script": script}
        options = [BG / r.background] if r.background else list(BG.glob("*.mp4"))
        if not options or not options[0].exists():
            raise RuntimeError("Add at least one .mp4 to the backgrounds folder.")
        await pipeline.render(script, r.voice, random.choice(options), OUT / job_id)
        jobs[job_id] = {"status": "done", "script": script, "url": f"/outputs/{job_id}/final.mp4"}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/generate")
async def generate(r: Req):
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "Queued"}
    asyncio.create_task(run(job_id, r))
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


@app.get("/api/backgrounds")
def backgrounds():
    return sorted(p.name for p in BG.glob("*.mp4"))


async def run_clip(job_id: str, src, url: str, n: int):
    d = OUT / job_id
    try:
        if url:
            jobs[job_id] = {"status": "Downloading video"}
            src = await clipper.download(url, d)
        jobs[job_id] = {"status": "Transcribing (long videos take a few minutes)"}
        segs, words, dur = await asyncio.to_thread(clipper.transcribe, src)
        jobs[job_id] = {"status": "Choosing the best moments"}
        clips = await asyncio.to_thread(clipper.pick_clips, segs, n, dur)
        done = []
        for i, c in enumerate(clips, 1):
            jobs[job_id] = {"status": f"Rendering clip {i} of {len(clips)}"}
            await clipper.render_clip(src, c, words, d, f"clip{i}")
            done.append({**c, "url": f"/outputs/{job_id}/clip{i}.mp4"})
        jobs[job_id] = {"status": "done", "clips": done}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/clip")
async def clip(file: UploadFile | None = File(None), url: str = Form(""),
               n_clips: int = Form(5), rights: bool = Form(False)):
    if not rights:
        raise HTTPException(400, "Confirm you own this video or have permission to clip it.")
    if not file and not url:
        raise HTTPException(400, "Upload a video or paste a link.")
    job_id = uuid.uuid4().hex[:12]
    d = OUT / job_id
    d.mkdir(parents=True)
    src = None
    if file:
        src = d / ("source" + Path(file.filename or "v.mp4").suffix)
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
    jobs[job_id] = {"status": "Queued"}
    asyncio.create_task(run_clip(job_id, src, "" if file else url, max(1, min(n_clips, 10))))
    return {"id": job_id}


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
