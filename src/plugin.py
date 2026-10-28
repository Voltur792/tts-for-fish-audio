from __future__ import annotations

import asyncio
import io
import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

from astra_plugin_sdk import (
    BadArguments,
    InternalError,
    NotConfigured,
    NotFound,
    Plugin,
    PluginError,
    RateLimited,
    Timeout,
    Unauthorized,
    Unavailable,
    VoiceInfo,
)

API_BASE = "https://api.fish.audio"
TTS_PATH = "/v1/tts"
MODELS_PATH = "/model"

DEFAULT_MODEL = "s2.1-pro-free"
MODEL_CHOICES = ("s2.1-pro", "s2.1-pro-free", "s2-pro", "s1", "drama-3")
DEFAULT_LATENCY = "normal"
LATENCY_CHOICES = ("normal", "balanced", "low")

DEFAULT_VOICE_ID = "default"
DEFAULT_VOICE_NAME = "Fish Audio default voice"
DEFAULT_VOICE_NAME_KEY = "voice.default"

AUDIO_FORMAT = "mp3"
SAMPLE_RATE = 44100
MP3_BITRATE = 128
MP3_BITRATE_CHOICES = (64, 128, 192)
SPEED_MIN = 0.5
SPEED_MAX = 2.0

TTS_TIMEOUT = 70.0
LISTING_TIMEOUT = 8.0
TRANSPORT_ATTEMPTS = 3
RETRY_BACKOFF_SECS = (0.5, 2.0)
RETRY_HTTP_MAX_WAIT = 5.0
MAX_AUDIO_BYTES = 3 * 1024 * 1024

VOICE_PAGE_SIZE = 50
OWN_PAGE_SIZE = 100
OWN_MAX_PAGES = 3
VOICE_CACHE_SECS = 300.0
LISTING_ATTEMPTS = 2

DOCS_URL = "https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech.md"
KEY_URL = "https://fish.audio/app/api-keys/"

log = logging.getLogger(__name__)


def _clamp_speed(value: object) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if speed != speed:
        return 1.0
    return max(SPEED_MIN, min(SPEED_MAX, speed))


def _is_default_pitch(pitch: object) -> bool:
    try:
        return abs(float(pitch) - 1.0) < 1e-3
    except (TypeError, ValueError):
        return True


def _first_string(values: object) -> str:
    if not isinstance(values, list):
        return ""
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _gender_for(item: dict) -> str:
    tags = {str(tag).lower() for tag in (item.get("tags") or [])}
    if tags & {"female", "women", "woman", "girl", "female voice"}:
        return "female"
    if tags & {"male", "men", "man", "boy", "male voice"}:
        return "male"
    return "neutral"


def _preview_for(item: dict) -> str:
    for sample in item.get("samples") or []:
        audio = sample.get("audio") if isinstance(sample, dict) else None
        if isinstance(audio, str) and audio.startswith(("http://", "https://")):
            return audio
    return ""


def _error_reason(body: bytes) -> str:
    if not body:
        return ""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body[:200].decode("utf-8", "replace").replace("\n", " ").strip()
    if not isinstance(data, dict):
        return ""
    return " — ".join(str(data[key]) for key in ("message", "reason") if data.get(key))


def _retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _is_retryable_http(status: int) -> bool:
    return status in (500, 502, 503, 504)


def _is_tls_failure(exc: OSError) -> bool:
    if isinstance(exc, ssl.SSLError):
        return True
    cause = exc.reason if isinstance(exc, urllib.error.URLError) else None
    return isinstance(cause, ssl.SSLError) or "[SSL:" in str(exc)


def _wav_duration_ms(audio: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio)) as wav:
            rate = wav.getframerate()
            return int(wav.getnframes() * 1000 / rate) if rate else 0
    except Exception:
        return 0


def _duration_ms(audio: bytes, audio_format: str, sample_rate: int, bitrate_kbps: int) -> int:
    if audio_format == "mp3" and bitrate_kbps > 0:
        return int(len(audio) * 8 / bitrate_kbps)
    if audio_format == "wav":
        return _wav_duration_ms(audio)
    if audio_format == "pcm" and sample_rate > 0:
        return int(len(audio) * 1000 / (sample_rate * 2))
    return 0


class FishTts(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self._voices: list[VoiceInfo] = []
        self._voices_at: float = 0.0
        self._warned_models: set[str] = set()
        self._pin_tls_12 = False

    async def on_config_changed(self, config: dict) -> None:
        self._voices = []
        self._voices_at = 0.0

    def _setting(self, name: str, default: str = "") -> str:
        value = (self.config or {}).get(name, default)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _api_key(self) -> str:
        return self._setting("api_key") or self._setting("tts_api_key")

    def _legacy_voice(self) -> str:
        return self._setting("voice_id")

    def _model(self) -> str:
        model = self._setting("model") or DEFAULT_MODEL
        if model not in MODEL_CHOICES and model not in self._warned_models:
            self._warned_models.add(model)
            log.warning(
                "unknown Fish Audio model %r, sending it through as-is: expected one of %s",
                model,
                ", ".join(MODEL_CHOICES),
            )
        return model

    def _latency(self) -> str:
        latency = self._setting("latency") or DEFAULT_LATENCY
        return latency if latency in LATENCY_CHOICES else DEFAULT_LATENCY

    def _bitrate(self) -> int:
        asked = self._setting("mp3_bitrate")
        if not asked:
            return MP3_BITRATE
        try:
            value = int(float(asked))
        except ValueError:
            log.warning("MP3 bitrate %r is not a number, keeping %d", asked, MP3_BITRATE)
            return MP3_BITRATE
        if value not in MP3_BITRATE_CHOICES:
            log.warning(
                "MP3 bitrate %d is not one of %s, keeping %d",
                value,
                ", ".join(str(option) for option in MP3_BITRATE_CHOICES),
                MP3_BITRATE,
            )
            return MP3_BITRATE
        return value

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _http(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = TTS_TIMEOUT,
        tls_1_2_only: bool = False,
    ) -> tuple[int, dict[str, str], bytes]:
        url = f"{API_BASE}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
        if data is not None:
            request.add_header("Content-Type", "application/json")
        context = None
        if tls_1_2_only:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return response.status, dict(response.headers), response.read()

    async def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = TTS_TIMEOUT,
        attempts: int = TRANSPORT_ATTEMPTS,
    ) -> tuple[int, dict[str, str], bytes]:
        pin = self._pin_tls_12
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.to_thread(
                    self._http,
                    method,
                    path,
                    params=params,
                    payload=payload,
                    headers=headers,
                    timeout=timeout,
                    tls_1_2_only=pin,
                )
            except urllib.error.HTTPError as exc:
                if not _is_retryable_http(exc.code) or attempt == attempts:
                    raise self._from_http_error(
                        exc.code, exc.headers, exc.read(), attempts=attempt
                    ) from None
                await self._backoff(self._retry_wait(exc.headers, attempt))
            except OSError as exc:
                if _is_tls_failure(exc):
                    if not pin:
                        pin = True
                        self._pin_tls_12 = True
                        log.warning(
                            "Fish Audio dropped the encrypted connection (%s); the next attempt pins TLS 1.2",
                            exc.__class__.__name__,
                        )
                if attempt == attempts:
                    raise self._from_transport_error(exc, timeout, attempts) from None
                log.warning(
                    "Fish Audio connection failed with %s, retrying (%d/%d)",
                    type(exc).__name__,
                    attempt,
                    attempts,
                )
                await self._backoff(RETRY_BACKOFF_SECS[min(attempt - 1, len(RETRY_BACKOFF_SECS) - 1)])
        raise InternalError("the retry loop ended without an answer", detail=path)

    async def _backoff(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def _retry_wait(self, headers: Any, attempt: int) -> float:
        asked = _retry_after(headers)
        if asked is not None and asked <= RETRY_HTTP_MAX_WAIT:
            return asked
        return RETRY_BACKOFF_SECS[min(attempt - 1, len(RETRY_BACKOFF_SECS) - 1)]

    def _from_transport_error(self, exc: OSError, timeout: float, attempts: int) -> PluginError:
        cause = exc.reason if isinstance(exc, urllib.error.URLError) and exc.reason else exc
        note = f"{type(cause).__name__}: {cause}" if cause is not exc else f"{type(exc).__name__}: {exc}"
        tried = f" after {attempts} attempts" if attempts > 1 else ""
        if isinstance(cause, (TimeoutError, socket.timeout)) or isinstance(exc, TimeoutError):
            return Timeout(
                f"Fish Audio did not answer within {int(timeout)} s{tried}",
                doc_url=DOCS_URL,
                detail=note,
            )
        if isinstance(cause, ssl.SSLError) or isinstance(exc, ssl.SSLError) or "[SSL:" in note:
            return Unavailable(
                f"the encrypted connection to Fish Audio was dropped{tried}: {note}. "
                "A VPN, an anti-virus or a proxy that decrypts HTTPS can do this; "
                "try again, or ask the assistant to answer in text.",
                doc_url=DOCS_URL,
                detail=note,
            )
        return Unavailable(
            f"network error talking to Fish Audio{tried}: {note}",
            doc_url=DOCS_URL,
            detail=note,
        )

    def _from_http_error(
        self, status: int, headers: Any, body: bytes, attempts: int = 1
    ) -> PluginError:
        reason = _error_reason(body)
        detail = f"HTTP {status}" + (f": {reason}" if reason else "")
        retried = f" (after {attempts} attempts)" if attempts > 1 else ""
        if status in (401, 403):
            return Unauthorized(
                reason or "Fish Audio refused this API key", doc_url=KEY_URL, detail=detail
            )
        if status == 402:
            return Unavailable(
                reason or "Fish Audio refused: this account cannot pay for speech",
                doc_url=KEY_URL,
                detail=detail,
            )
        if status == 404:
            return NotFound(reason or "that voice does not exist", detail=detail)
        if status == 429:
            return RateLimited(
                reason or "Fish Audio rate limit reached",
                retry_after=_retry_after(headers),
                detail=detail,
            )
        if _is_retryable_http(status):
            return Unavailable(
                (reason or "Fish Audio is temporarily unavailable") + retried, detail=detail
            )
        return BadArguments(reason or f"Fish Audio rejected the request ({status})", detail=detail)

    def _default_voice(self) -> VoiceInfo:
        name = self.i18n.t(DEFAULT_VOICE_NAME_KEY)
        if name == DEFAULT_VOICE_NAME_KEY:
            name = DEFAULT_VOICE_NAME
        return VoiceInfo(id=DEFAULT_VOICE_ID, name=name, language="", gender="neutral")

    def _to_voice(self, item: Any) -> VoiceInfo | None:
        if not isinstance(item, dict):
            return None
        model_id = item.get("_id")
        if not model_id or item.get("state") not in (None, "trained"):
            return None
        return VoiceInfo(
            id=str(model_id),
            name=str(item.get("title") or model_id),
            language=_first_string(item.get("languages")),
            gender=_gender_for(item),
            preview_url=_preview_for(item),
        )

    def _items_of(self, body: bytes) -> list[Any]:
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise InternalError("Fish Audio returned an unreadable voice list", detail=str(exc)) from None
        items = data.get("items") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []

    async def _list_page(self, params: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return await self._call(
            "GET",
            MODELS_PATH,
            params=params,
            headers=self._auth_headers(),
            timeout=LISTING_TIMEOUT,
            attempts=LISTING_ATTEMPTS,
        )

    async def _fetch_own_voices(self) -> list[VoiceInfo]:
        voices: list[VoiceInfo] = []
        for page in range(1, OWN_MAX_PAGES + 1):
            _, _, body = await self._list_page(
                {"self": "true", "page_size": OWN_PAGE_SIZE, "page_number": page}
            )
            items = self._items_of(body)
            for item in items:
                voice = self._to_voice(item)
                if voice is not None:
                    voices.append(voice)
            if len(items) < OWN_PAGE_SIZE:
                break
        return voices

    async def _fetch_library_voices(self) -> list[VoiceInfo]:
        _, _, body = await self._list_page(
            {"page_size": VOICE_PAGE_SIZE, "page_number": 1, "sort_by": "score"}
        )
        return [voice for voice in map(self._to_voice, self._items_of(body)) if voice is not None]

    async def _fetch_voices(self) -> list[VoiceInfo]:
        collected: dict[str, VoiceInfo] = {}
        for voice in await self._fetch_own_voices():
            collected[voice.id] = voice
        for voice in await self._fetch_library_voices():
            collected.setdefault(voice.id, voice)
        return list(collected.values())

    async def tts_list_voices(self) -> list[VoiceInfo]:
        voices = [self._default_voice()]
        if not self._api_key():
            return voices

        cached = list(self._voices)
        if cached and time.monotonic() - self._voices_at < VOICE_CACHE_SECS:
            return voices + cached

        try:
            fetched = await self._fetch_voices()
        except PluginError as exc:
            log.warning("voice listing failed: %s", exc.message)
            return voices + cached

        self._voices = fetched
        self._voices_at = time.monotonic()
        return voices + fetched

    async def tts_synthesize(
        self, text: str, voice_id: str, speed: float, pitch: float
    ) -> dict[str, Any]:
        api_key = self._api_key()
        if not api_key:
            raise NotConfigured("api_key", doc_url=KEY_URL)
        if not text or not text.strip():
            raise BadArguments("nothing to say: the text was empty")

        if not _is_default_pitch(pitch):
            log.debug(
                "Fish Audio has no pitch control, ignoring pitch=%r for voice %r", pitch, voice_id
            )

        bitrate = self._bitrate()
        reference_id = None if voice_id in ("", DEFAULT_VOICE_ID) else voice_id
        if reference_id is None and voice_id == "":
            reference_id = self._legacy_voice() or None

        payload: dict[str, Any] = {
            "text": text,
            "reference_id": reference_id,
            "format": AUDIO_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "mp3_bitrate": bitrate,
            "normalize": True,
            "chunk_length": 300,
            "latency": self._latency(),
            "prosody": {"speed": _clamp_speed(speed), "normalize_loudness": True},
        }
        language = self._setting("language")
        if language:
            payload["language"] = language

        headers = self._auth_headers() | {"model": self._model()}
        _, _, audio = await self._call("POST", TTS_PATH, payload=payload, headers=headers)

        if not audio:
            raise InternalError(
                "Fish Audio answered with no audio", detail="empty 2xx body", doc_url=DOCS_URL
            )
        if len(audio) > MAX_AUDIO_BYTES:
            raise Unavailable(
                "the reply is too long to synthesize in one request",
                doc_url=DOCS_URL,
                detail=f"{len(audio)} bytes of audio were generated",
            )

        log.debug("synthesized %d characters for voice %r at %d kbps", len(text), voice_id, bitrate)
        return {
            "audio_data": audio,
            "format": AUDIO_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "duration_ms": _duration_ms(audio, AUDIO_FORMAT, SAMPLE_RATE, bitrate),
        }


if __name__ == "__main__":
    FishTts().run()
