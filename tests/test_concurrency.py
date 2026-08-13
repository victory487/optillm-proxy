from __future__ import annotations

import pytest

from bon_proxy.concurrency import ConcurrencyGate


@pytest.mark.anyio
async def test_gate_rejects_without_queueing_and_releases_slot() -> None:
    gate = ConcurrencyGate(1)

    assert await gate.try_acquire() is True
    assert gate.active == 1
    assert await gate.try_acquire() is False
    assert gate.active == 1

    await gate.release()

    assert gate.active == 0
    assert await gate.try_acquire() is True
    await gate.release()


@pytest.mark.anyio
async def test_slot_releases_after_exception() -> None:
    gate = ConcurrencyGate(1)

    with pytest.raises(RuntimeError, match="boom"):
        async with gate.slot() as acquired:
            assert acquired is True
            raise RuntimeError("boom")

    assert gate.active == 0
