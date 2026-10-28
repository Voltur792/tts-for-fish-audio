"""Tests for the TTS for Fish Audio plugin.

Level 1 (in process, no daemon, through the real gRPC servicer): `Harness`
covers what the daemon actually sees — registration, the voice picker, and the
error plane of `tts_synthesize`.

The Fish API is never called: `_http` is replaced with a recording stub, so the
assertions are about the request we send and the errors we raise, which is what
a plugin can get wrong on its own side.

Run: `pytest`.
"""

import asyncio
import io
import json
import ssl
import sys
import urllib.error
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from astra_plugin_sdk import (  # noqa: E402
    BadArguments,
    InternalError,
    NotFound,
    RateLimited,
    Timeout,
    Unauthorized,
    Unavailable,
)
from astra_plugin_sdk.testing import Harness, HarnessError, fuzz_configs  # noqa: E402

from src.plugin import (  # noqa: E402
    AUDIO_FORMAT,
    DEFAULT_LATENCY,
    DEFAULT_MODEL,
    DEFAULT_VOICE_ID,
    MP3_BITRATE,
    LISTING_ATTEMPTS,
    OWN_PAGE_SIZE,
    SAMPLE_RATE,
    TRANSPORT_ATTEMPTS,
    VOICE_PAGE_SIZE,
    FishTts,
    _duration_ms,
)

API_KEY = "test-api-key"
FAKE_AUDIO = b"ID3" + b"\x00" * 15_997


def run(coroutine):
    return asyncio.run(coroutine)


async def no_backoff(seconds: float) -> None:
    """Tests never sleep between retries; they only count the attempts."""


def plugin(**settings) -> FishTts:
    instance = FishTts()
    instance.config = dict(settings)
    instance._backoff = no_backoff
    return instance


class StubHttp:
    """Replaces `FishTts._http`: records every request, replays scripted answers."""

    def __init__(self, answers) -> None:
        self.answers = list(answers)
        self.requests: list[dict] = []

    def __call__(self, method, path, *, params=None, payload=None, headers=None, timeout=None, **transport):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "payload": payload,
                "headers": dict(headers or {}),
                "timeout": timeout,
                "transport": transport,
            }
        )
        answer = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        status, body, extra = answer
        return status, dict(extra or {}), body

    @property
    def last(self) -> dict:
        return self.requests[-1]


def stub(instance: FishTts, *answers) -> StubHttp:
    instance._http = StubHttp(answers)
    return instance._http


def http_error(code: int, message: str, headers: dict | None = None) -> urllib.error.HTTPError:
    body = json.dumps({"message": message}).encode()
    return urllib.error.HTTPError("https://api.fish.audio/v1/tts", code, message, headers or {}, io.BytesIO(body))


def synthesize(instance: FishTts, text="hello", voice_id=DEFAULT_VOICE_ID, speed=1.0, pitch=1.0):
    return run(instance.tts_synthesize(text, voice_id, speed, pitch))


# ── what the daemon sees ──


def test_starts_and_answers_a_health_check():
    with Harness(FishTts()) as harness:
        healthy, _status = harness.health()
        assert healthy


def test_the_voice_page_offers_the_model_voice_before_any_key_exists():
    # A fresh install has no key, and it still has to be selectable and
    # configurable: an empty voice list is what makes a plugin drop out of the
    # capability report entirely.
    with Harness(FishTts()) as harness:
        assert [voice.id for voice in harness.voices()] == [DEFAULT_VOICE_ID]


def test_a_missing_key_is_reported_as_the_field_to_fill_in():
    with Harness(FishTts()) as harness:
        with pytest.raises(HarnessError) as failure:
            harness.tts("hello")
        assert "NOT_CONFIGURED" in str(failure.value)
        assert "api_key" in str(failure.value)


def test_a_full_synthesis_survives_the_real_grpc_route():
    instance = FishTts()
    instance._http = StubHttp([(200, FAKE_AUDIO, None)])
    with Harness(instance).with_config({"api_key": API_KEY}) as harness:
        response = harness.tts("hello", voice_id=DEFAULT_VOICE_ID)
        assert response.audio_data == FAKE_AUDIO
        assert response.format == AUDIO_FORMAT
        assert response.sample_rate == SAMPLE_RATE
        assert response.duration_ms == int(len(FAKE_AUDIO) * 8 / MP3_BITRATE)


def test_no_config_the_daemon_can_deliver_never_crashes_this_plugin():
    # `{}` is the first payload every plugin ever sees; `fuzz_configs()` adds the
    # user's typing and an older version of this schema. None of it may throw.
    with Harness(FishTts()) as harness:
        for payload in fuzz_configs():
            harness.set_config(payload)
        assert [voice.id for voice in harness.voices()] == [DEFAULT_VOICE_ID]


# ── what we send to Fish ──


def test_the_synthesis_request_matches_the_fish_contract():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, (200, FAKE_AUDIO, None))

    result = synthesize(instance, text="Привет.")

    request = transport.last
    assert request["method"] == "POST"
    assert request["path"] == "/v1/tts"
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["headers"]["model"] == DEFAULT_MODEL

    body = request["payload"]
    assert body["text"] == "Привет."
    assert body["reference_id"] is None
    assert body["prosody"]["speed"] == 1.0
    assert body["latency"] == DEFAULT_LATENCY
    assert body["format"] == AUDIO_FORMAT

    assert result["audio_data"] == FAKE_AUDIO
    assert result["format"] == AUDIO_FORMAT
    assert result["sample_rate"] == SAMPLE_RATE
    assert result["duration_ms"] == int(len(FAKE_AUDIO) * 8 / MP3_BITRATE)


def test_a_selected_voice_becomes_the_reference_id():
    instance = plugin(api_key=API_KEY, model="s2-pro", latency="low")
    transport = stub(instance, (200, FAKE_AUDIO, None))

    synthesize(instance, voice_id="voice-model-id", speed=0.8)

    request = transport.last
    assert request["payload"]["reference_id"] == "voice-model-id"
    assert request["payload"]["prosody"]["speed"] == 0.8
    assert request["payload"]["latency"] == "low"
    assert request["headers"]["model"] == "s2-pro"


def test_the_key_and_voice_left_by_version_0_1_keep_working_after_upgrade():
    instance = plugin(tts_api_key=API_KEY, voice_id="vm-legacy")
    transport = stub(instance, (200, FAKE_AUDIO, None))
    result = synthesize(instance, voice_id="")
    assert transport.last["payload"]["reference_id"] == "vm-legacy"
    assert result["audio_data"] == FAKE_AUDIO


def test_an_explicitly_selected_default_voice_ignores_the_legacy_setting():
    instance = plugin(api_key=API_KEY, voice_id="vm-legacy")
    transport = stub(instance, (200, FAKE_AUDIO, None))
    synthesize(instance, voice_id=DEFAULT_VOICE_ID)
    assert transport.last["payload"]["reference_id"] is None


def test_speed_is_clamped_to_the_range_fish_accepts():
    for given, expected in ((2.5, 2.0), (0.1, 0.5), ("nonsense", 1.0), (1.4, 1.4)):
        instance = plugin(api_key=API_KEY)
        transport = stub(instance, (200, FAKE_AUDIO, None))
        synthesize(instance, speed=given)
        assert transport.last["payload"]["prosody"]["speed"] == expected


def test_pitch_is_ignored_because_fish_has_no_pitch_control():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, (200, FAKE_AUDIO, None))
    synthesize(instance, pitch=1.7)
    assert "pitch" not in transport.last["payload"]


def test_language_is_sent_only_when_the_user_asked_for_it():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, (200, FAKE_AUDIO, None))
    synthesize(instance)
    assert "language" not in transport.last["payload"]

    hinted = plugin(api_key=API_KEY, language="ru")
    transport = stub(hinted, (200, FAKE_AUDIO, None))
    synthesize(hinted)
    assert transport.last["payload"]["language"] == "ru"


def test_the_bitrate_the_user_offers_reaches_the_request_and_the_duration():
    instance = plugin(api_key=API_KEY, mp3_bitrate="64")
    transport = stub(instance, (200, FAKE_AUDIO, None))
    result = synthesize(instance)
    assert transport.last["payload"]["mp3_bitrate"] == 64
    assert result["duration_ms"] == int(len(FAKE_AUDIO) * 8 / 64)

    default = plugin(api_key=API_KEY)
    transport = stub(default, (200, FAKE_AUDIO, None))
    result = synthesize(default)
    assert transport.last["payload"]["mp3_bitrate"] == MP3_BITRATE
    assert result["duration_ms"] == int(len(FAKE_AUDIO) * 8 / MP3_BITRATE)


def test_empty_text_is_an_argument_error_and_sends_no_request():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, (200, FAKE_AUDIO, None))
    with pytest.raises(BadArguments):
        synthesize(instance, text="   ")
    assert transport.requests == []


# ── what we raise for what Fish answers ──


def test_a_refused_key_is_unauthorized_not_a_bare_failure():
    instance = plugin(api_key="nope")
    stub(instance, http_error(401, "invalid api key"))
    with pytest.raises(Unauthorized):
        synthesize(instance)


def test_an_unknown_voice_is_not_found():
    instance = plugin(api_key=API_KEY)
    stub(instance, http_error(404, "voice not found"))
    with pytest.raises(NotFound):
        synthesize(instance, voice_id="missing-id")


def test_a_rate_limit_carries_the_retry_hint():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, http_error(429, "slow down", {"Retry-After": "7"}))
    with pytest.raises(RateLimited) as failure:
        synthesize(instance)
    assert failure.value.retry_after == 7.0
    assert len(transport.requests) == 1


def test_a_rate_limit_says_nothing_about_retrying_it_internally():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, http_error(429, "slow down", {"Retry-After": "1"}))
    with pytest.raises(RateLimited):
        synthesize(instance)
    assert len(transport.requests) == 1


# a dropped TLS record is what this plugin most often hits in practice: a VPN, an
# anti-virus or a proxy that decrypts HTTPS. It is worth one quiet retry.


def test_a_dropped_tls_connection_is_retried_and_then_answers():
    instance = plugin(api_key=API_KEY)
    transport = stub(
        instance,
        OSError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac"),
        (200, FAKE_AUDIO, None),
    )
    result = synthesize(instance)
    assert len(transport.requests) == 2
    assert result["audio_data"] == FAKE_AUDIO


def test_a_tls_failure_pins_tls_1_2_on_the_next_attempt():
    # An HTTPS-scanning VPN, anti-virus or proxy that breaks TLS 1.3 is the usual
    # producer of DECRYPTION_FAILED_OR_BAD_RECORD_MAC: the retry after one must
    # not negotiate 1.3 again.
    instance = plugin(api_key=API_KEY)
    transport = stub(
        instance,
        ssl.SSLError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac"),
        (200, FAKE_AUDIO, None),
    )
    result = synthesize(instance)
    assert len(transport.requests) == 2
    assert transport.requests[0]["transport"]["tls_1_2_only"] is False
    assert transport.requests[1]["transport"]["tls_1_2_only"] is True
    assert result["audio_data"] == FAKE_AUDIO


def test_a_pinned_tls_1_2_is_kept_for_the_whole_session():
    # The middlebox does not go away between calls: every later request starts
    # pinned, so none of them burns its first attempt on the default handshake.
    instance = plugin(api_key=API_KEY)
    transport = stub(
        instance,
        ssl.SSLError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac"),
        (200, FAKE_AUDIO, None),
        (200, FAKE_AUDIO, None),
        (200, FAKE_AUDIO, None),
    )
    synthesize(instance)
    assert transport.requests[1]["transport"]["tls_1_2_only"] is True
    result = synthesize(instance)
    assert len(transport.requests) == 3
    assert transport.requests[2]["transport"]["tls_1_2_only"] is True
    assert result["audio_data"] == FAKE_AUDIO


def test_a_repeatedly_dropped_tls_connection_names_the_failure_and_says_it_retried():
    instance = plugin(api_key=API_KEY)
    transport = stub(
        instance,
        ssl.SSLError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac"),
    )
    with pytest.raises(Unavailable) as failure:
        synthesize(instance)
    assert len(transport.requests) == TRANSPORT_ATTEMPTS
    assert failure.value.config_field == ""
    assert f"after {TRANSPORT_ATTEMPTS} attempts" in failure.value.message
    assert "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in failure.value.message
    assert "VPN" in failure.value.message


def test_an_unrelated_network_failure_is_not_blamed_on_tls():
    instance = plugin(api_key=API_KEY)
    stub(instance, ConnectionResetError("connection reset by peer"))
    with pytest.raises(Unavailable) as failure:
        synthesize(instance)
    assert "ConnectionResetError" in failure.value.message
    assert "VPN" not in failure.value.message


def test_an_overloaded_upstream_is_unavailable():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, http_error(503, "high load"))
    with pytest.raises(Unavailable):
        synthesize(instance)
    assert len(transport.requests) == TRANSPORT_ATTEMPTS


def test_an_overloaded_upstream_is_retried_and_then_answers():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, http_error(503, "high load"), (200, FAKE_AUDIO, None))
    result = synthesize(instance)
    assert len(transport.requests) == 2
    assert result["audio_data"] == FAKE_AUDIO


def test_a_network_failure_is_unavailable_not_a_crash():
    instance = plugin(api_key=API_KEY)
    stub(instance, urllib.error.URLError("no route to fish.audio"))
    with pytest.raises(Unavailable):
        synthesize(instance)


def test_a_slow_answer_is_reported_as_a_timeout():
    instance = plugin(api_key=API_KEY)
    stub(instance, TimeoutError())
    with pytest.raises(Timeout):
        synthesize(instance)


def test_a_stalled_handshake_is_reported_as_a_timeout():
    instance = plugin(api_key=API_KEY)
    stub(instance, urllib.error.URLError(TimeoutError("timed out")))
    with pytest.raises(Timeout):
        synthesize(instance)


def test_an_empty_body_is_an_internal_error():
    instance = plugin(api_key=API_KEY)
    stub(instance, (200, b"", None))
    with pytest.raises(InternalError):
        synthesize(instance)


def test_a_reply_too_long_for_one_request_is_reported():
    instance = plugin(api_key=API_KEY)
    stub(instance, (200, b"\x00" * (4 * 1024 * 1024), None))
    with pytest.raises(Unavailable):
        synthesize(instance)


# ── the voice picker ──


def test_the_voice_page_lists_the_accounts_own_voices_then_the_library():
    instance = plugin(api_key=API_KEY)
    own = {
        "items": [
            {
                "_id": "vm1",
                "title": "Alice",
                "state": "trained",
                "languages": ["en"],
                "tags": ["female"],
                "samples": [{"audio": "https://cdn.fish.audio/vm1.mp3"}],
            },
            {"_id": "vm2", "title": "Broken", "state": "failed"},
            {"_id": "", "title": "No id", "state": "trained"},
        ]
    }
    library = {
        "items": [
            {
                "_id": "vm3",
                "title": "Боб",
                "state": "trained",
                "languages": ["ru"],
                "tags": ["male"],
                "samples": [{"audio": "/relative/not-a-url"}],
            },
            {"_id": "vm1", "title": "Alice again", "state": "trained"},
        ]
    }
    transport = stub(instance, (200, json.dumps(own), None), (200, json.dumps(library), None))

    offered = run(instance.tts_list_voices())

    own_call, library_call = transport.requests
    assert own_call["params"]["self"] == "true"
    assert own_call["params"]["page_size"] == OWN_PAGE_SIZE
    assert "self" not in library_call["params"]
    assert library_call["params"]["page_size"] == VOICE_PAGE_SIZE

    assert [voice.id for voice in offered] == [DEFAULT_VOICE_ID, "vm1", "vm3"]
    assert offered[1].name == "Alice"
    assert offered[1].language == "en"
    assert offered[1].gender == "female"
    assert offered[1].preview_url == "https://cdn.fish.audio/vm1.mp3"
    assert offered[-1].name == "Боб"
    assert offered[-1].gender == "male"
    assert offered[-1].preview_url == ""


def test_the_voice_page_also_offers_a_voice_that_is_not_in_the_library():
    # A voice a user made themselves is usually buried under a thousand public
    # ones, and `self=true` is the only call that finds it.
    instance = plugin(api_key=API_KEY)
    own = {"items": [{"_id": "root-voice", "title": "Root", "state": "trained", "languages": ["ru"]}]}
    library = {"items": [{"_id": "public-one", "title": "Public", "state": "trained"}]}
    stub(instance, (200, json.dumps(own), None), (200, json.dumps(library), None))

    offered = [voice.id for voice in run(instance.tts_list_voices())]
    assert offered == [DEFAULT_VOICE_ID, "root-voice", "public-one"]


def test_voice_listing_is_cached_and_refreshed_on_config_change():
    instance = plugin(api_key=API_KEY)
    own = json.dumps({"items": [{"_id": "vm1", "title": "x"}]})
    library = json.dumps({"items": []})
    transport = stub(instance, (200, own, None), (200, library, None))

    run(instance.tts_list_voices())
    run(instance.tts_list_voices())
    assert len(transport.requests) == 2

    run(instance.on_config_changed({"api_key": "other"}))
    run(instance.tts_list_voices())
    assert len(transport.requests) == 4


def test_a_failed_voice_listing_never_empties_the_voice_page():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, http_error(401, "invalid api key"))
    assert [voice.id for voice in run(instance.tts_list_voices())] == [DEFAULT_VOICE_ID]
    assert len(transport.requests) == 1

    without_key = plugin()
    assert [voice.id for voice in run(without_key.tts_list_voices())] == [DEFAULT_VOICE_ID]


def test_a_listing_that_fails_on_tls_gets_the_pinned_attempt_too():
    instance = plugin(api_key=API_KEY)
    transport = stub(instance, ssl.SSLError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] bad record mac"))
    assert [voice.id for voice in run(instance.tts_list_voices())] == [DEFAULT_VOICE_ID]
    assert len(transport.requests) == LISTING_ATTEMPTS
    assert transport.requests[-1]["transport"]["tls_1_2_only"] is True


# ── duration math ──


def test_duration_matches_the_container_we_ask_for():
    assert _duration_ms(b"\x00" * 16_000, "mp3", SAMPLE_RATE, MP3_BITRATE) == 1000
    assert _duration_ms(b"\x00" * (SAMPLE_RATE * 2), "pcm", SAMPLE_RATE, 0) == 1000

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x00" * (24_000 * 2 * 3))
    assert _duration_ms(buffer.getvalue(), "wav", SAMPLE_RATE, 0) == 1000 * 3


# ── the settings page and the names the user reads ──


def test_the_default_voice_name_follows_the_users_language(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    with Harness(FishTts()).with_language("ru") as harness:
        names = [voice.name for voice in harness.voices()]
    assert "Голос модели Fish Audio" in names

    with Harness(FishTts()).with_language("en") as harness:
        names = [voice.name for voice in harness.voices()]
    assert "Fish Audio model voice" in names


def test_every_field_the_manifest_declares_has_a_title_and_description():
    import tomllib

    manifest = tomllib.loads((REPO_ROOT / "plugin.toml").read_text(encoding="utf-8"))
    schema = json.loads(manifest["config"]["schema"])
    fields = schema["properties"]
    for path in (REPO_ROOT / "locales" / "en.json", REPO_ROOT / "locales" / "ru.json"):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        assert all(isinstance(value, str) for value in catalog.values()), path
        for field_id in fields:
            assert catalog.get(f"config.{field_id}.title"), (path, field_id)
            assert catalog.get(f"config.{field_id}.description"), (path, field_id)
        assert catalog.get("voice.default"), path


def test_the_russian_catalog_is_keyed_against_the_english_one():
    en = json.loads((REPO_ROOT / "locales" / "en.json").read_text(encoding="utf-8"))
    ru = json.loads((REPO_ROOT / "locales" / "ru.json").read_text(encoding="utf-8"))
    missing = sorted(set(ru) - set(en))
    assert missing == []
