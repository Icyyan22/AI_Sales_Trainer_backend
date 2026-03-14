from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from passlib.hash import pbkdf2_sha256
from pydantic import BaseModel, Field
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.models.db import UserRecord, async_session_factory

router = APIRouter(tags=["auth"])

SECRET_KEY = os.environ.get("TRAINER_JWT_SECRET", "ai-sales-trainer-secret-key-2024")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "avatars")
os.makedirs(UPLOAD_DIR, exist_ok=True)

security = HTTPBearer(auto_error=False)


# ---------- Schemas ----------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=30)
    password: str = Field(..., min_length=4, max_length=100)
    display_name: str = Field("", max_length=50)


class LoginRequest(BaseModel):
    username: str
    password: str


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., pattern=r"^(user|admin)$")


class UserResponse(BaseModel):
    id: str
    username: str
    display_name: str
    avatar_url: str | None
    role: str
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


# ---------- Helpers ----------

def _create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _user_to_response(user: UserRecord) -> UserResponse:
    avatar_url = f"/api/v1/auth/avatar/{user.id}" if user.avatar_path else None
    return UserResponse(
        id=user.id,
        username=user.username,
        display_name=user.display_name or user.username,
        avatar_url=avatar_url,
        role=user.role or "user",
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserRecord | None:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            return None
    except JWTError:
        return None

    async with async_session_factory() as db:
        user = await db.get(UserRecord, user_id)
        return user


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserRecord:
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


async def require_admin(
    user: UserRecord = Depends(require_user),
) -> UserRecord:
    if user.role not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


async def require_super_admin(
    user: UserRecord = Depends(require_user),
) -> UserRecord:
    if user.role != "super_admin":
        raise HTTPException(status_code=403, detail="需要超级管理员权限")
    return user


# ---------- Endpoints ----------

@router.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    async with async_session_factory() as db:
        existing = await db.execute(
            select(UserRecord).where(UserRecord.username == req.username)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="用户名已存在")

        # Determine role: first user or matching env config → super_admin
        role = "user"
        if settings.super_admin_username and req.username == settings.super_admin_username:
            role = "super_admin"
        else:
            count_result = await db.execute(select(sa_func.count()).select_from(UserRecord))
            total_users = count_result.scalar() or 0
            if total_users == 0:
                role = "super_admin"

        user = UserRecord(
            id=str(uuid.uuid4()),
            username=req.username,
            password_hash=pbkdf2_sha256.hash(req.password),
            display_name=req.display_name or req.username,
            role=role,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    token = _create_token(user.id)
    return AuthResponse(token=token, user=_user_to_response(user))


@router.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    async with async_session_factory() as db:
        result = await db.execute(
            select(UserRecord).where(UserRecord.username == req.username)
        )
        user = result.scalar_one_or_none()

    if not user or not pbkdf2_sha256.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _create_token(user.id)
    return AuthResponse(token=token, user=_user_to_response(user))


@router.get("/auth/me", response_model=UserResponse)
async def get_me(user: UserRecord = Depends(require_user)):
    return _user_to_response(user)


@router.put("/auth/profile", response_model=UserResponse)
async def update_profile(
    req: UpdateProfileRequest,
    user: UserRecord = Depends(require_user),
):
    async with async_session_factory() as db:
        db_user = await db.get(UserRecord, user.id)
        if not db_user:
            raise HTTPException(status_code=404, detail="用户不存在")
        if req.display_name is not None:
            db_user.display_name = req.display_name
        await db.commit()
        await db.refresh(db_user)
        return _user_to_response(db_user)


@router.post("/auth/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    user: UserRecord = Depends(require_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持上传图片文件")

    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过2MB")

    ext = file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "png"
    filename = f"{user.id}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    async with async_session_factory() as db:
        db_user = await db.get(UserRecord, user.id)
        if db_user:
            db_user.avatar_path = filename
            await db.commit()
            await db.refresh(db_user)
            return _user_to_response(db_user)

    raise HTTPException(status_code=404, detail="用户不存在")


@router.get("/auth/avatar/{user_id}")
async def get_avatar(user_id: str):
    async with async_session_factory() as db:
        user = await db.get(UserRecord, user_id)

    if not user or not user.avatar_path:
        raise HTTPException(status_code=404, detail="头像不存在")

    filepath = os.path.join(UPLOAD_DIR, user.avatar_path)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="头像文件不存在")

    return FileResponse(filepath)


# ---------- User Management (admin+) ----------

@router.get("/auth/users")
async def list_users(user: UserRecord = Depends(require_admin)):
    async with async_session_factory() as db:
        result = await db.execute(
            select(UserRecord).order_by(UserRecord.created_at.desc())
        )
        users = result.scalars().all()
        return {"users": [_user_to_response(u) for u in users]}


@router.put("/auth/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    req: UpdateRoleRequest,
    current_user: UserRecord = Depends(require_super_admin),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能修改自己的角色")

    async with async_session_factory() as db:
        target = await db.get(UserRecord, user_id)
        if not target:
            raise HTTPException(status_code=404, detail="用户不存在")
        if target.role == "super_admin":
            raise HTTPException(status_code=400, detail="不能修改超级管理员角色")
        target.role = req.role
        await db.commit()
        await db.refresh(target)
        return _user_to_response(target)
