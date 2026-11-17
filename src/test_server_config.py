"""Server config composition: LZAI_SERVER_CONFIG lists, extends, merge rules, !replace."""

import json

import pytest

from src import server_config


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def prod(tmp_path):
    """A shared base and one fragment per GPU, as on the home box."""
    _write(tmp_path / "base.yml", """
engines: {pipertts: true, voxcpm: true, seed_vc: true}
voxcpm:
  model_path: data/lzspeech-falcon-v1.2
  inference_timesteps: 16
  applicable_loras: {accent-en-GB: data/a, accent-en-US: data/b}
  default_locales: [en-US, vi-VN]
load: {mode: all}
""")
    _write(tmp_path / "workers/3090.yml", """
extends: ../base.yml
transcribe: {enabled: true}
media: {lipsync: true, videogen: true}
load: {mode: on-demand, budget_gb: 22, pinned: [aligner]}
voxcpm:
  inference_timesteps: null
  default_locales: [en-US]
""")
    return tmp_path


def test_extends_merges_mappings_and_replaces_everything_else(prod):
    data, files = server_config.load(str(prod / "workers/3090.yml"))
    assert files == [prod / "base.yml", prod / "workers/3090.yml"]
    assert data["engines"] == {"pipertts": True, "voxcpm": True, "seed_vc": True}
    assert data["load"] == {"mode": "on-demand", "budget_gb": 22, "pinned": ["aligner"]}
    # Keys the fragment leaves out stay; a list replaces; null is a value (it does not delete the key).
    assert data["voxcpm"] == {
        "model_path": "data/lzspeech-falcon-v1.2",
        "inference_timesteps": None,
        "applicable_loras": {"accent-en-GB": "data/a", "accent-en-US": "data/b"},
        "default_locales": ["en-US"],
    }
    assert "extends" not in data


def test_a_comma_list_merges_left_to_right(prod, monkeypatch):
    _write(prod / "local.yml", "voxcpm: {model_path: data/lzspeech-falcon-v1.1}\nload: {budget_gb: 20}\n")
    monkeypatch.setenv("LZAI_SERVER_CONFIG", f"{prod / 'workers/3090.yml'}, {prod / 'local.yml'}")
    data, files = server_config.load()
    assert files[-1] == prod / "local.yml"
    assert data["voxcpm"]["model_path"] == "data/lzspeech-falcon-v1.1"
    assert data["load"] == {"mode": "on-demand", "budget_gb": 20, "pinned": ["aligner"]}


def test_replace_tag_drops_the_bases_mapping(prod):
    _write(prod / "v100.yml", "extends: base.yml\nvoxcpm:\n  applicable_loras: !replace {ipa-text: data/c}\n")
    data, _ = server_config.load(str(prod / "v100.yml"))
    assert data["voxcpm"]["applicable_loras"] == {"ipa-text": "data/c"}
    assert data["voxcpm"]["model_path"] == "data/lzspeech-falcon-v1.2"
    assert type(data["voxcpm"]["applicable_loras"]) is dict


def test_several_bases_merge_in_order(tmp_path):
    _write(tmp_path / "a.yml", "x: {p: 1, q: 1}\n")
    _write(tmp_path / "b.yml", "x: {q: 2, r: 2}\n")
    _write(tmp_path / "c.yml", "extends: [a.yml, b.yml]\nx: {r: 3}\n")
    data, files = server_config.load(str(tmp_path / "c.yml"))
    assert data == {"x": {"p": 1, "q": 2, "r": 3}}
    assert [f.name for f in files] == ["a.yml", "b.yml", "c.yml"]


def test_json_still_loads(tmp_path):
    path = _write(tmp_path / "server.json", json.dumps({"engines": {"voxcpm": True}, "voxcpm": {"ipa_adapter_path": None}}))
    assert server_config.load(str(path))[0] == {"engines": {"voxcpm": True}, "voxcpm": {"ipa_adapter_path": None}}


def test_default_is_the_checkouts_yml_then_its_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LZAI_SERVER_CONFIG", raising=False)
    assert server_config.load() == ({}, [])
    _write(tmp_path / "local/server.json", '{"a": 1}')
    assert server_config.load()[0] == {"a": 1}
    _write(tmp_path / "local/server.yml", "a: 2\n")
    assert server_config.load()[0] == {"a": 2}


def test_missing_files_cycles_and_bad_shapes_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.yml"):
        server_config.load(str(tmp_path / "missing.yml"))
    _write(tmp_path / "child.yml", "extends: nope.yml\n")
    with pytest.raises(FileNotFoundError, match="extended by"):
        server_config.load(str(tmp_path / "child.yml"))
    _write(tmp_path / "a.yml", "extends: b.yml\n")
    _write(tmp_path / "b.yml", "extends: a.yml\n")
    with pytest.raises(ValueError, match="extends itself"):
        server_config.load(str(tmp_path / "a.yml"))
    _write(tmp_path / "list.yml", "- a\n- b\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        server_config.load(str(tmp_path / "list.yml"))
    _write(tmp_path / "bad.yml", "extends: {a: 1}\n")
    with pytest.raises(ValueError, match="path or a list"):
        server_config.load(str(tmp_path / "bad.yml"))


def test_worker_flags_keys_that_are_not_settings():
    from src.api.server import ServerConfig, _unknown_config_keys

    data = {"voxcpm": {"inference_timesteps": 16, "inference_timestep": 12}, "lod": {}, "engines": {"voxcpm": True},
            "voxcpm_typo": 1}
    assert _unknown_config_keys(ServerConfig, data) == ["voxcpm.inference_timestep", "lod", "voxcpm_typo"]
