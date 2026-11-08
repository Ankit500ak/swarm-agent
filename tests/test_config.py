import json

from swarm_agent.config import Config, load_config, normalize_url


def test_defaults():
    cfg = Config()
    assert cfg.model == "qwen2.5-coder:7b"
    assert cfg.peers == ["http://localhost:11434"]
    assert cfg.auto_discover is False


def test_normalize_url():
    assert normalize_url("localhost:11434") == "http://localhost:11434"
    assert normalize_url("http://192.168.1.20:11434/") == "http://192.168.1.20:11434"
    assert normalize_url("  http://x:1 ") == "http://x:1"


def test_env_overrides(tmp_path):
    cfg = load_config(
        config_path=tmp_path / "missing.json",
        env={"SWARM_MODEL": "qwen2.5-coder:14b",
             "SWARM_PEERS": "http://a:11434, 192.168.1.9:11434"},
    )
    assert cfg.model == "qwen2.5-coder:14b"
    assert cfg.peers == ["http://a:11434", "http://192.168.1.9:11434"]


def test_ollama_host_is_primary_peer(tmp_path):
    cfg = load_config(
        config_path=tmp_path / "missing.json",
        env={"SWARM_PEERS": "http://a:11434", "OLLAMA_HOST": "192.168.1.50:11434"},
    )
    assert cfg.peers[0] == "http://192.168.1.50:11434"
    assert "http://a:11434" in cfg.peers
    assert len(cfg.peers) == 2  # no duplication of the default peer


def test_file_then_env_wins(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "model": "file-model",
        "peers": ["http://filehost:11434"],
        "chunk_tokens": 777,
        "unknown_key": "ignored",
    }))
    cfg = load_config(config_path=path, env={})
    assert cfg.model == "file-model"
    assert cfg.peers == ["http://filehost:11434"]
    assert cfg.chunk_tokens == 777

    cfg2 = load_config(config_path=path, env={"SWARM_MODEL": "env-model"})
    assert cfg2.model == "env-model"
    assert cfg2.chunk_tokens == 777  # file value survives where env is silent
