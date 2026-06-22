from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class Session(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    title: str
    status: str = "new"

    # --- Metadaten (NEU) ---
    doctor_name: str = ""
    patient_name: str = ""
    patient_dob: str = ""          # als String "YYYY-MM-DD" (einfach fürs MVP)
    case_id: str = ""              # Fallnummer
    department: str = ""           # Station/Abteilung
    language_hint: str = ""        # optional "de", "en", ...

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "staff"
    created_at: datetime = Field(default_factory=datetime.utcnow)
