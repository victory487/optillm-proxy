from __future__ import annotations

import logging

import httpx
import pytest

from bon_proxy.app import create_app
from tests.test_api import RecordingTransport, basic_request


@pytest.mark.anyio
async def test_logs_do_not_include_prompts_or_api_keys(app_config, caplog) -> None:
    caplog.set_level(logging.INFO)
    answer = RecordingTransport([httpx.ConnectError("offline")])
    judge = RecordingTransport([])
    app = create_app(app_config, answer_transport=answer, judge_transport=judge)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=basic_request(messages=[{"role": "user", "content": "highly-sensitive-prompt"}]),
        )

    assert response.status_code == 502
    assert "highly-sensitive-prompt" not in caplog.text
    assert "answer-secret" not in caplog.text
    assert "judge-secret" not in caplog.text
