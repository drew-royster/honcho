from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

DEFAULT_CODEX_AUTH_FILE = Path("~/.codex/auth.json").expanduser()
DEFAULT_CODEX_TOKEN_ENDPOINT = "https://auth0.openai.com/oauth/token"


class CodexAuthError(RuntimeError):
    """Raised when Codex OAuth credentials are missing or invalid."""


def _relogin_error(detail: str) -> CodexAuthError:
    return CodexAuthError(f"{detail} Run `codex` and sign in again.")


class CodexAuthStore:
    """Load and refresh Codex OAuth credentials from ~/.codex/auth.json."""

    def __init__(
        self,
        auth_file: str | os.PathLike[str] | None = None,
        *,
        token_endpoint: str = DEFAULT_CODEX_TOKEN_ENDPOINT,
        refresh_buffer_seconds: int = 300,
        http_client_factory: Callable[[], Any] | None = None,
    ):
        self._auth_file = Path(auth_file).expanduser() if auth_file else DEFAULT_CODEX_AUTH_FILE
        self._token_endpoint = token_endpoint
        self._refresh_buffer_seconds = refresh_buffer_seconds
        self._http_client_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=10.0)
        )
        self._lock = asyncio.Lock()

    @property
    def auth_file(self) -> Path:
        return self._auth_file

    async def get_access_token(self) -> str:
        async with self._lock:
            auth_payload = self._load_auth_payload()
            access_token = self._get_token(auth_payload, "access_token")

            if self._is_token_expiring_soon(access_token):
                auth_payload = await self._refresh_auth_payload(auth_payload)
                access_token = self._get_token(auth_payload, "access_token")

            return access_token

    def _load_auth_payload(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._auth_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise _relogin_error(
                f"Codex OAuth auth file was not found at `{self._auth_file}`."
            ) from exc
        except json.JSONDecodeError as exc:
            raise _relogin_error(
                f"Codex OAuth auth file at `{self._auth_file}` is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise _relogin_error(
                f"Codex OAuth auth file at `{self._auth_file}` has an invalid top-level shape."
            )

        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            raise _relogin_error(
                f"Codex OAuth auth file at `{self._auth_file}` is missing `tokens`."
            )

        self._get_token(payload, "access_token")
        self._get_token(payload, "refresh_token")
        self._get_client_id(self._get_token(payload, "access_token"))
        return payload

    def _get_token(self, payload: dict[str, Any], key: str) -> str:
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            raise _relogin_error(
                f"Codex OAuth auth file at `{self._auth_file}` is missing `tokens`."
            )

        value = tokens.get(key)
        if not isinstance(value, str) or not value:
            raise _relogin_error(
                f"Codex OAuth auth file at `{self._auth_file}` is missing `tokens.{key}`."
            )
        return value

    def _is_token_expiring_soon(self, access_token: str) -> bool:
        payload = self._decode_jwt_payload(access_token)
        expires_at = payload.get("exp")
        if not isinstance(expires_at, int):
            raise _relogin_error(
                f"Codex OAuth access token in `{self._auth_file}` is missing an `exp` claim."
            )

        now = datetime.now(timezone.utc).timestamp()
        return expires_at <= now + self._refresh_buffer_seconds

    def _get_client_id(self, access_token: str) -> str:
        payload = self._decode_jwt_payload(access_token)
        client_id = payload.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            raise _relogin_error(
                f"Codex OAuth access token in `{self._auth_file}` is missing `client_id`."
            )
        return client_id

    def _decode_jwt_payload(self, token: str) -> dict[str, Any]:
        try:
            _header, payload_b64, _signature = token.split(".", 2)
        except ValueError as exc:
            raise _relogin_error(
                f"Codex OAuth access token in `{self._auth_file}` is malformed."
            ) from exc

        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        try:
            payload_bytes = base64.urlsafe_b64decode(padded.encode("utf-8"))
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise _relogin_error(
                f"Codex OAuth access token in `{self._auth_file}` could not be decoded."
            ) from exc

        if not isinstance(payload, dict):
            raise _relogin_error(
                f"Codex OAuth access token in `{self._auth_file}` decoded to an invalid payload."
            )

        return payload

    async def _refresh_auth_payload(self, auth_payload: dict[str, Any]) -> dict[str, Any]:
        refresh_token = self._get_token(auth_payload, "refresh_token")
        client_id = self._get_client_id(self._get_token(auth_payload, "access_token"))

        request_body = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        }

        try:
            async with self._http_client_factory() as client:
                response = await client.post(self._token_endpoint, json=request_body)
                response.raise_for_status()
                refreshed = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _relogin_error(
                f"Refreshing Codex OAuth tokens from `{self._auth_file}` failed."
            ) from exc

        if not isinstance(refreshed, dict):
            raise _relogin_error("Codex OAuth refresh returned an invalid payload.")

        new_access_token = refreshed.get("access_token")
        if not isinstance(new_access_token, str) or not new_access_token:
            raise _relogin_error("Codex OAuth refresh did not return an access token.")

        updated_payload = deepcopy(auth_payload)
        tokens = updated_payload.setdefault("tokens", {})
        if not isinstance(tokens, dict):
            raise _relogin_error("Codex OAuth auth payload has an invalid `tokens` block.")

        tokens["access_token"] = new_access_token

        new_refresh_token = refreshed.get("refresh_token")
        if isinstance(new_refresh_token, str) and new_refresh_token:
            tokens["refresh_token"] = new_refresh_token

        new_id_token = refreshed.get("id_token")
        if isinstance(new_id_token, str) and new_id_token:
            tokens["id_token"] = new_id_token

        updated_payload["last_refresh"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        self._write_auth_payload(updated_payload)
        return updated_payload

    def _write_auth_payload(self, payload: dict[str, Any]) -> None:
        self._auth_file.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            dir=self._auth_file.parent,
            prefix=f".{self._auth_file.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self._auth_file)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
