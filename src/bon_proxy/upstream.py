"""Minimal raw-JSON client for OpenAI-compatible vLLM endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from bon_proxy.config import UpstreamConfig
from bon_proxy.errors import UpstreamHTTPError, UpstreamProtocolError, UpstreamTimeout


class VLLMClient:
    def __init__(
        self,
        name: str,
        config: UpstreamConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.config = config
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(
            base_url=f"{config.base_url}/",
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(config.timeout_seconds),
            trust_env=False,
        )

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self._client.post("chat/completions", json=payload)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise UpstreamTimeout(self.name) from exc
        except httpx.HTTPError as exc:
            raise UpstreamHTTPError(self.name, 0) from exc

        if not response.is_success:
            raise UpstreamHTTPError(self.name, response.status_code)

        try:
            data = response.json()
        except ValueError as exc:
            raise UpstreamProtocolError(self.name, "response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise UpstreamProtocolError(self.name, "response JSON must be an object")
        return data

    async def close(self) -> None:
        await self._client.aclose()
