from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session

router = APIRouter()


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict:
    db_ok = True
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
