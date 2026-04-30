"""Per-client rate limiting via Redis tumbling-window counters.

Keyed on `ClientApp.id` (not IP) so the same client gets the same budget across
sources. Bucket key includes a minute index, so each minute starts fresh. The
counter expires 65s after creation — slightly past the bucket window so a late
read still sees the right count.

Unlimited if `client.rate_limit_per_minute` is null.
"""

from __future__ import annotations

import time

from fastapi import Depends, HTTPException, status

from auth import current_client
from models.client import ClientApp
from worker.queue import get_redis

KEY_PREFIX = "conduct:ratelimit"
WINDOW_S = 60


async def rate_limited_client(
    client: ClientApp = Depends(current_client),
) -> ClientApp:
    if client.rate_limit_per_minute is None:
        return client

    now = int(time.time())
    bucket = now // WINDOW_S
    key = f"{KEY_PREFIX}:{client.id}:{bucket}"

    redis = get_redis()
    count = redis.incr(key)
    if count == 1:
        # First hit in this bucket — set TTL so old buckets don't accumulate.
        redis.expire(key, WINDOW_S + 5)

    if count > client.rate_limit_per_minute:
        retry_after = WINDOW_S - (now % WINDOW_S)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded ({client.rate_limit_per_minute}/min)",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(client.rate_limit_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(now + retry_after),
            },
        )

    return client
