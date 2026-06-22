from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlmodel import Session as DBSession, select

from db import engine
from models import User

# ✅ Argon2 ist stabiler als bcrypt (und klinik-/enterprise-tauglich)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ⚠️ ÄNDERN! (nur DEV-Secret, später via ENV)
JWT_SECRET = "PLEASE_CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_>=_32_CHARS"
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = 8 * 60  # 8h

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(sub: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)
    payload = {"sub": sub, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_user_by_username(username: str) -> Optional[User]:
    with DBSession(engine) as s:
        return s.exec(select(User).where(User.username == username)).first()


def ensure_admin_seeded() -> None:
    """
    Erststart: legt einen Admin an, wenn noch keiner existiert.
    Default-PW NUR fürs DEV – danach sofort ändern.
    """
    with DBSession(engine) as s:
        existing = s.exec(select(User).where(User.username == "admin")).first()
        if existing:
            return
        admin = User(username="admin", password_hash=hash_password("admin1234"), role="admin")
        s.add(admin)
        s.commit()


def require_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(*roles: str):
    def dep(user: User = Depends(require_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dep
