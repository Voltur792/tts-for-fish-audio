"""Level 2 (§5.6): the plugin's real startup path over loopback gRPC.

A level-1 `Harness` cannot substitute for this test: it runs `Plugin._run_async`,
so the registration handshake, the capability auth interceptor and the protobuf
descriptors the plugin actually sends are all real. It also proves the plugin
answers its own capability report without touching the network — the voice list
is what `tts` is advertised on, and no API key exists at registration time.
"""

from astra_plugin_sdk.proto import plugin_pb2
from astra_plugin_sdk.testing import WireHarness

from src.plugin import DEFAULT_VOICE_ID, FishTts

AUDIO_BYTES = b"\xff\xfb\x90\x64" + b"\x00" * 3_000


def test_the_plugin_registers_and_answers_on_the_wire():
    with WireHarness(FishTts()) as wire:
        assert wire.health().healthy

        registered = wire.daemon.registrations[-1]
        assert list(registered.capabilities) == ["tts"]

        voices = wire.stub().TtsListVoices(plugin_pb2.Empty(), metadata=wire.metadata)
        assert [voice.id for voice in voices.voices] == [DEFAULT_VOICE_ID]


def test_audio_comes_back_over_the_wire_as_real_protobuf_bytes():
    instance = FishTts()
    instance._http = lambda method, path, **kwargs: (200, {}, AUDIO_BYTES)
    with WireHarness(instance, config={"api_key": "wire-key"}) as wire:
        response = wire.stub().TtsSynthesize(
            plugin_pb2.PluginTtsSynthesizeRequest(text="hello", voice_id=DEFAULT_VOICE_ID),
            metadata=wire.metadata,
        )
        assert response.audio_data == AUDIO_BYTES
        assert response.format == "mp3"
        assert response.sample_rate
        assert response.duration_ms > 0
