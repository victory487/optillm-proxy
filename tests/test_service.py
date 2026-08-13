from __future__ import annotations

import json

import pytest

from bon_proxy.errors import ProxyError
from bon_proxy.service import BestOfNService, Candidate
from tests.helpers import answer_response, choice, judge_response


class StubClient:
    async def chat_completions(self, payload):  # pragma: no cover - not used directly
        raise AssertionError(payload)

    async def close(self) -> None:
        return None


@pytest.fixture
def service(app_config) -> BestOfNService:
    return BestOfNService(app_config, StubClient(), StubClient())  # type: ignore[arg-type]


def test_answer_payload_overrides_controlled_fields_and_preserves_extras(service) -> None:
    body = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "top_p": 0.2,
        "n": 99,
        "chat_template_kwargs": {"thinking": False},
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_object"},
        "custom_vllm_field": 7,
    }

    payload = service._build_answer_payload(body)

    assert payload["model"] == "answer-model"
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["n"] == 3
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "max"
    assert payload["stream"] is False
    assert payload["return_token_ids"] is True
    assert payload["tools"] == body["tools"]
    assert payload["response_format"] == body["response_format"]
    assert payload["custom_vllm_field"] == 7
    assert body["model"] == "client-model"


def test_judge_payload_contains_full_choices_without_token_ids(service) -> None:
    candidates = [
        Candidate(choice(0, "first", [1, 2]), [1, 2]),
        Candidate(choice(1, "second", [3, 4, 5]), [3, 4, 5]),
    ]
    request_body = {
        "model": "virtual",
        "messages": [{"role": "user", "content": "question"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    payload = service._build_judge_payload(request_body, candidates)

    assert payload["model"] == "judge-model"
    assert payload["temperature"] == 0.1
    assert payload["n"] == 1
    assert payload["chat_template_kwargs"] == {"thinking": False}
    assert (
        payload["response_format"]["json_schema"]["schema"]["properties"]["best_index"]["maximum"]
        == 1
    )
    judge_text = payload["messages"][1]["content"]
    assert isinstance(judge_text, str)
    assert '"tools"' in judge_text
    assert '"content":"second"' in judge_text
    assert "token_ids" not in judge_text


def test_judge_payload_preserves_media_as_content_parts(service) -> None:
    image_part = {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
    request_body = {
        "model": "virtual",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe"}, image_part],
            }
        ],
    }
    candidates = [
        Candidate(choice(0, "first", [1]), [1]),
        Candidate(choice(1, "second", [2]), [2]),
    ]

    payload = service._build_judge_payload(request_body, candidates)

    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[1] == image_part
    assert "_media_reference" in content[0]["text"]
    assert "https://example.test/a.png" not in content[0]["text"]


@pytest.mark.parametrize(
    "payload",
    [
        judge_response(1),
        {
            "choices": [
                {"message": {"content": [{"type": "output_text", "text": '{"best_index":1}'}]}}
            ]
        },
    ],
)
def test_parse_judge_index(payload) -> None:
    assert BestOfNService._parse_judge_index(payload, 2) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "not-json"}}]},
        {"choices": [{"message": {"content": '{"best_index":true}'}}]},
        {"choices": [{"message": {"content": '{"best_index":2}'}}]},
    ],
)
def test_parse_judge_index_rejects_invalid_output(payload) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        BestOfNService._parse_judge_index(payload, 2)


def test_final_response_preserves_selected_choice_and_recomputes_usage() -> None:
    selected = choice(
        1,
        None,
        [9, 8, 7],
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id":1}'},
            }
        ],
    )
    upstream = answer_response([choice(0, "first", [1]), selected])

    response = BestOfNService._build_final_response(upstream, Candidate(selected, [9, 8, 7]))

    assert len(response["choices"]) == 1
    assert response["choices"] == [selected]
    assert response["choices"][0]["index"] == 1
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert "prompt_token_ids" not in response
    assert response["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }
    assert "completion_tokens_details" not in response["usage"]
    assert "token_ids" in upstream["choices"][1]


def test_final_response_requires_token_ids() -> None:
    selected = choice(0, "answer", [])
    selected.pop("token_ids")

    with pytest.raises(ProxyError) as exc_info:
        BestOfNService._build_final_response(answer_response([selected]), Candidate(selected, None))

    assert exc_info.value.code == "missing_token_ids"


def test_final_response_without_token_ids_preserves_sglang_usage_and_message() -> None:
    selected = choice(1, "selected answer", [])
    selected.pop("token_ids")
    selected["message"]["reasoning_content"] = "private reasoning"
    upstream = answer_response([choice(0, "first", [1, 2]), selected])
    upstream["usage"] = {
        "prompt_tokens": 11,
        "completion_tokens": 9,
        "total_tokens": 20,
        "reasoning_tokens": 4,
    }

    response = BestOfNService._build_final_response(
        upstream,
        Candidate(selected, None),
        require_token_ids=False,
    )

    assert response["choices"] == [selected]
    assert response["choices"][0]["message"]["content"] == "selected answer"
    assert response["usage"] == upstream["usage"]
