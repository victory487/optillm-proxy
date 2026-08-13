from __future__ import annotations

import copy

import pytest
import yaml

from bon_proxy.config import AppConfig, ConfigLoadError, load_config


def test_load_valid_yaml(tmp_path, config_dict) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    config = load_config(path)

    assert config.answer.params.n == 3
    assert config.judge.params.n == 1
    assert config.server.log_level == "INFO"
    assert config.answer.base_url == "http://answer.test/v1"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("server", "max_concurrency"), 0),
        (("answer", "timeout_seconds"), 0),
        (("answer", "params", "n"), 1),
        (("judge", "params", "n"), 2),
        (("answer", "params", "top_p"), 0),
        (("answer", "params", "temperature"), 2.1),
    ],
)
def test_config_rejects_invalid_values(config_dict, path, value) -> None:
    data = copy.deepcopy(config_dict)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        AppConfig.model_validate(data)


def test_config_rejects_unknown_fields(config_dict) -> None:
    config_dict["server"]["workers"] = 2

    with pytest.raises(ValueError):
        AppConfig.model_validate(config_dict)


def test_load_config_reports_invalid_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("server: [", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="invalid YAML"):
        load_config(path)


def test_load_config_requires_mapping(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- item", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="YAML mapping"):
        load_config(path)
