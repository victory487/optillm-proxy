"""OpenAI-compatible proxy errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProxyError(Exception):
    status_code: int
    message: str
    error_type: str
    code: str
    param: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class UpstreamError(Exception):
    """Base class for upstream transport and protocol failures."""


class UpstreamTimeout(UpstreamError):
    def __init__(self, upstream: str) -> None:
        self.upstream = upstream
        super().__init__(f"{upstream} upstream timed out")


class UpstreamHTTPError(UpstreamError):
    def __init__(self, upstream: str, status_code: int) -> None:
        self.upstream = upstream
        self.status_code = status_code
        super().__init__(f"{upstream} upstream returned HTTP {status_code}")


class UpstreamProtocolError(UpstreamError):
    def __init__(self, upstream: str, detail: str) -> None:
        self.upstream = upstream
        self.detail = detail
        super().__init__(f"{upstream} upstream protocol error: {detail}")
