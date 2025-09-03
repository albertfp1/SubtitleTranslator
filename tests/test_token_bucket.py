import time
import pytest
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from translation_client import TokenBucket


def test_token_bucket_waits_between_acquires():
    tb = TokenBucket(capacity=1, refill_rate=1)  # 1 token per second

    async def run_test():
        await tb.acquire()
        start = time.monotonic()
        await tb.acquire()  # should wait ~1 second
        elapsed1 = time.monotonic() - start

        start = time.monotonic()
        await tb.acquire()
        elapsed2 = time.monotonic() - start
        return elapsed1, elapsed2

    elapsed1, elapsed2 = asyncio.run(run_test())
    assert elapsed1 >= 1.0
    assert elapsed2 >= 1.0
