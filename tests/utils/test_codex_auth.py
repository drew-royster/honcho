import json
import tempfile
from pathlib import Path

import pytest

from src.utils.codex_auth import CodexAuthStore


def _make_jwt(*, client_id: str = "codex-cli", exp: int = 4_102_444_800) -> str:
    header = {"alg": "none", "typ": "JWT"}
    payload = {"client_id": client_id, "exp": exp}

    def _encode(data: dict[str, object]) -> str:
        import base64

        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_encode(header)}.{_encode(payload)}."


class FakeHTTPClient:
    def __init__(self, payload: dict[str, object], calls: list[dict[str, object]]):
        self._payload = payload
        self._calls = calls

    async def __aenter__(self) -> "FakeHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> bool:
        return False

    async def post(self, url: str, json: dict[str, object]):
        self._calls.append({"url": url, "json": json})
        return FakeHTTPResponse(self._payload)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.mark.asyncio
async def test_codex_auth_store_loads_valid_access_token_without_refresh():
    with tempfile.TemporaryDirectory() as temp_dir:
        auth_path = Path(temp_dir) / "auth.json"
        token = _make_jwt()
        auth_path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": token,
                        "refresh_token": "refresh-token",
                    }
                }
            ),
            encoding="utf-8",
        )

        store = CodexAuthStore(auth_file=auth_path, refresh_buffer_seconds=0)
        assert await store.get_access_token() == token


@pytest.mark.asyncio
async def test_codex_auth_store_refreshes_expiring_access_token():
    with tempfile.TemporaryDirectory() as temp_dir:
        auth_path = Path(temp_dir) / "auth.json"
        expired_token = _make_jwt(exp=1)
        refreshed_token = _make_jwt(exp=4_202_444_800)
        auth_path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": expired_token,
                        "refresh_token": "refresh-before",
                    }
                }
            ),
            encoding="utf-8",
        )

        calls: list[dict[str, object]] = []
        store = CodexAuthStore(
            auth_file=auth_path,
            http_client_factory=lambda: FakeHTTPClient(
                {
                    "access_token": refreshed_token,
                    "refresh_token": "refresh-after",
                },
                calls,
            ),
        )

        assert await store.get_access_token() == refreshed_token

        written = json.loads(auth_path.read_text(encoding="utf-8"))
        assert written["tokens"]["access_token"] == refreshed_token
        assert written["tokens"]["refresh_token"] == "refresh-after"
        assert calls[0]["json"]["refresh_token"] == "refresh-before"
