"""Model set selection (LZAI_MODELS / LZAI_REQUIRED_MODELS) and ready task types."""

import json

import pytest

from src import model_sets
from src.taskflow_worker import ready_task_types, served_task_types


@pytest.fixture
def server_json(tmp_path, monkeypatch):
    path = tmp_path / "server.json"
    path.write_text(json.dumps({"engines": {"voxcpm": True, "seed_vc": False}, "transcribe": {"enabled": True}, "media": {"music": True}}))
    monkeypatch.setenv("LZAI_SERVER_CONFIG", str(path))
    monkeypatch.delenv("LZAI_MODELS", raising=False)
    monkeypatch.delenv("LZAI_REQUIRED_MODELS", raising=False)
    return path


def test_default_is_what_server_json_enables_and_all_required(server_json):
    assert model_sets.selected_sets() == ["transcribe", "falcon", "music"]
    assert model_sets.required_sets() == ["transcribe", "falcon", "music"]
    assert model_sets.explicit_selection() is None
    assert model_sets.media_stage() == "required"


def test_required_subset_defers_the_rest(server_json, monkeypatch):
    monkeypatch.setenv("LZAI_MODELS", "falcon, music")
    monkeypatch.setenv("LZAI_REQUIRED_MODELS", "falcon")
    assert model_sets.selected_sets() == ["falcon", "music"]
    assert model_sets.required_sets() == ["falcon"]
    assert model_sets.media_stage() == "deferred"
    monkeypatch.setenv("LZAI_REQUIRED_MODELS", "none")
    assert model_sets.required_sets() == []


def test_required_outside_models_and_unknown_names_fail(server_json, monkeypatch):
    monkeypatch.setenv("LZAI_MODELS", "falcon")
    monkeypatch.setenv("LZAI_REQUIRED_MODELS", "transcribe")
    with pytest.raises(ValueError, match="not in LZAI_MODELS"):
        model_sets.required_sets()
    monkeypatch.setenv("LZAI_MODELS", "falcon,sparrow")
    with pytest.raises(ValueError, match="unknown model set"):
        model_sets.selected_sets()
    monkeypatch.setenv("LZAI_MODELS", "all")
    assert model_sets.selected_sets() == list(model_sets.MODEL_SETS)


def test_dirs_map_to_sets_and_unclaimed_dirs_are_core():
    assert model_sets.set_of_dir("lzspeech-falcon-v1.2") == {"falcon"}
    assert model_sets.set_of_dir("voxcpm2-stable") == {"falcon"}
    assert model_sets.set_of_dir("comfyui") == {"imagegen", "videogen", "lipsync"}
    assert model_sets.set_of_dir("span-upscale") == {"videogen", "lipsync"}
    assert model_sets.set_of_dir("qwen3-asr-0.6b") == {"transcribe"}
    for core in ("lzspeech-sparrow", "lzspeech-starling", "runtime-wheels", "number-modes", "voice-presets.json"):
        assert model_sets.set_of_dir(core) == set()


def test_pending_markers(tmp_path):
    assert model_sets.pending_sets(tmp_path) == set()
    model_sets.pending_dir(tmp_path).mkdir()
    (model_sets.pending_dir(tmp_path) / "falcon").touch()
    assert model_sets.pending_sets(tmp_path) == {"falcon"}


def test_types_wait_for_their_deferred_engines():
    types = ("tts-synthesis", "voice-design", "voice-enhance", "transcribe", "force-align", "videogen")
    assert ready_task_types(types, frozenset()) == types
    assert ready_task_types(types, frozenset({"voxcpm"})) == ("voice-enhance", "transcribe", "force-align", "videogen")
    assert ready_task_types(types, frozenset({"transcribe", "media"})) == ("tts-synthesis", "voice-design", "voice-enhance", "force-align")


def test_lzai_models_turns_engines_and_their_types_off(server_json, monkeypatch):
    from src.api import server

    monkeypatch.setenv("LZAI_MODELS", "falcon")
    config = server._load_config()
    assert config.engines.voxcpm and not config.engines.seed_vc
    assert not config.transcribe.enabled and config.media.features() == []
    served = served_task_types(config)
    assert "voice-design" in served and "transcribe" not in served and "generate-music" not in served
    assert "force-align" in served and "tts-synthesis" in served


def test_media_only_worker_serves_no_tts(monkeypatch):
    from src.api import server

    # The Salad pilot's worker: tracked base + the manifest's inline settings.
    monkeypatch.setenv("LZAI_SERVER_CONFIG", "configs/server/base.yml")
    monkeypatch.setenv(
        "LZAI_SERVER_CONFIG_YAML",
        "{engines: {pipertts: false, voxcpm: false, seed_vc: false}, voxcpm: {enabled: false},"
        " transcribe: {enabled: false}, media: {music: false, videogen: true, lipsync: true, imagegen: true},"
        " load: {mode: on-demand, budget_gb: 22, pinned: [aligner], preload: [media]}}",
    )
    monkeypatch.setenv("LZAI_MODELS", "videogen,lipsync,imagegen")
    monkeypatch.delenv("LZAI_LOAD_MODE", raising=False)
    monkeypatch.delenv("LZAI_VRAM_BUDGET_GB", raising=False)
    config = server._load_config()
    assert config.media.gpu is None and config.media.features() == ["videogen", "lipsync", "imagegen"]
    served = served_task_types(config)
    assert not {"tts-synthesis", "voice-design", "transcribe", "generate-music"} & set(served)
    assert {"videogen", "lipsync", "imagegen", "force-align"} <= set(served)


def test_falcon_downloads_only_what_server_json_serves(server_json, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_PATH", raising=False)
    assert model_sets.falcon_paths() == [model_sets.DEFAULT_FALCON_MODEL.removeprefix("data/")]
    raw = json.loads(server_json.read_text())
    raw["voxcpm"] = {
        "model_path": "./data/lzspeech-falcon-v1.2",
        "applicable_loras": {"ipa-text": "/srv/lzai/data/voxcpm-lora/ipa/ipa-text-falcon"},
    }
    server_json.write_text(json.dumps(raw))
    falcon = model_sets.falcon_paths()
    assert falcon == ["lzspeech-falcon-v1.2", "voxcpm-lora/ipa/ipa-text-falcon"]
    wanted = lambda rel, sets=("falcon",): model_sets.wanted_path(rel, sets, falcon)  # noqa: E731
    assert wanted("lzspeech-falcon-v1.2/model.safetensors")
    assert wanted("voxcpm-lora/ipa/ipa-text-falcon/lora_weights.safetensors")
    assert not wanted("lzspeech-falcon-v1.1/model.safetensors")
    assert not wanted("lzspeech-falcon-v1.2-old/model.safetensors")
    assert not wanted("voxcpm2-original/model.safetensors")
    assert not wanted("voxcpm-lora/accent/accent-en-US-falcon/lora_weights.safetensors")
    assert wanted("lzspeech-sparrow/model.ckpt") and wanted("comfyui/x", {"videogen"})
    assert not wanted("lzspeech-falcon-v1.2/model.safetensors", ("transcribe",))


def test_join_pool_metadata(monkeypatch):
    from src.taskflow_worker import pool_metadata

    monkeypatch.delenv("TASKFLOW_WORKER_POOL", raising=False)
    monkeypatch.delenv("TASKFLOW_WORKER_CAPACITY", raising=False)
    assert pool_metadata() == {"capacity": 1.0}
    monkeypatch.setenv("TASKFLOW_WORKER_POOL", "owned")
    monkeypatch.setenv("TASKFLOW_WORKER_CAPACITY", "1.8")
    assert pool_metadata() == {"pool": "owned", "capacity": 1.8}
    for bad in ("fast", "-1", "nan"):
        monkeypatch.setenv("TASKFLOW_WORKER_CAPACITY", bad)
        with pytest.raises(ValueError, match="TASKFLOW_WORKER_CAPACITY"):
            pool_metadata()
