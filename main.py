from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import shutil

from jobs import create_job, get_job, job_to_dict, start_transcribe_job, UPLOAD_DIR
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session as DBSession, select

from db import engine, init_db
from models import Session, User
from auth import (
    verify_password,
    create_access_token,
    require_user,
    ensure_admin_seeded,
)

class LoginRequest(BaseModel):
    username: str
    password: str

app = FastAPI(title="Clinic Assist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    ensure_admin_seeded()

@app.get("/health")
def health():
    return {"ok": True, "service": "clinic-assist-backend"}

@app.post("/auth/login")
def login(body: LoginRequest):
    with DBSession(engine) as s:
        user = s.exec(select(User).where(User.username == body.username)).first()
        if not user or not verify_password(body.password, user.password_hash):
            return {"ok": False, "message": "Invalid credentials"}
        token = create_access_token(sub=user.username, role=user.role)
        return {"ok": True, "access_token": token, "token_type": "bearer"}

@app.get("/auth/me")
def me(user: User = Depends(require_user)):
    return {"username": user.username, "role": user.role}

@app.get("/sessions", response_model=list[Session])
def list_sessions(user: User = Depends(require_user)):
    with DBSession(engine) as session:
        return session.exec(select(Session).order_by(Session.created_at.desc())).all()

@app.post("/sessions", response_model=Session)
def create_session(title: str, user: User = Depends(require_user)):
    with DBSession(engine) as session:
        s = Session(title=title)
        session.add(s)
        session.commit()
        session.refresh(s)
        return s

@app.post("/jobs/transcribe")
def transcribe_upload(
    file: UploadFile = File(...),
    with_timestamps: bool = Form(False),
    diarize_speakers: bool = Form(False),
    user: User = Depends(require_user),
):
    ext = os.path.splitext(file.filename or "")[1].lower()

    allowed = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".dss", ".ds2"]
    if ext not in allowed:
        return {"ok": False, "message": "Unsupported file type"}

    fname = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)

    # Große Dateien nicht komplett in den RAM laden.
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)

    job = create_job(kind="transcribe", input_path=path)
    start_transcribe_job(job.id, with_timestamps=with_timestamps, diarize_speakers=diarize_speakers)

    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, user: User = Depends(require_user)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job_to_dict(job)}

@app.get("/jobs/{job_id}/result.txt")
def download_result_txt(job_id: str, user: User = Depends(require_user)):
    job = get_job(job_id)
    if not job or not job.result_txt_path:
        raise HTTPException(status_code=404, detail="TXT export not available")
    return FileResponse(
        job.result_txt_path,
        media_type="text/plain; charset=utf-8",
        filename=f"{job_id}.txt"
    )


@app.get("/jobs/{job_id}/result.json")
def download_result_json(job_id: str, user: User = Depends(require_user)):
    job = get_job(job_id)
    if not job or not job.result_json_path:
        raise HTTPException(status_code=404, detail="JSON export not available")
    return FileResponse(
        job.result_json_path,
        media_type="application/json; charset=utf-8",
        filename=f"{job_id}.json"
    )
