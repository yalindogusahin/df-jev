import json

import httpx
import pandas as pd
import pytest

from df_jev import JevClient, JevError, JevFrame


def response(value=0.9):
    return {"model": "test-1", "answers": {"result": {"type": "noul", "noul": value}}}


def make_client(handler, **kwargs):
    return JevClient(
        "http://test", "dummy", "test-1", transport=httpx.MockTransport(handler), **kwargs
    )


def test_duplicate_indices_missing_values_and_selected_fields():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=response(0.9 if payload["state"]["message"] else 0.1))

    df = pd.DataFrame(
        {
            "message": ["locked out", None, "resolved"],
            "date": [pd.Timestamp("2026-01-01")] * 3,
            "secret": ["never send"] * 3,
        },
        index=[4, 4, 9],
    )
    with make_client(handler) as client:
        result = JevFrame(df, client).evaluate(
            "Current access failure", columns=["message", "date"]
        )
    assert result.index.tolist() == [4, 4, 9]
    assert result.match_probability.tolist() == [0.9, 0.1, 0.9]
    assert requests[1]["state"]["message"] is None
    assert requests[0]["state"]["date"].startswith("2026-01-01")
    assert all("secret" not in r["state"] for r in requests)
    assert list(df.columns) == ["message", "date", "secret"]


def test_cache_reuses_decisions_but_not_different_questions_or_data():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response())

    with make_client(handler) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a", "a", "b"]}), client)
        first = frame.evaluate("Question one", columns=["message"])
        assert first.match_cached.tolist() == [False, True, False]
        second = frame.filter("Question one", columns=["message"], threshold=0.95)
        assert second.empty
        assert len(calls) == 2
        frame.evaluate("Question two", columns=["message"])
        assert len(calls) == 4
        client.clear_cache()
        frame.evaluate("Question one", columns=["message"])
        assert len(calls) == 6


def test_invalid_responses_remain_failed_rows_and_are_not_cached():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(1.8))

    with make_client(handler) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a", None]}), client)
        result = frame.evaluate("Condition", columns=["message"])
        assert result.match_probability.isna().all()
        assert result.match_error.notna().all()
        assert len(calls) == 1  # Empty row never goes to the model.
        with pytest.raises(JevError, match="Some rows failed"):
            frame.filter("Condition", columns=["message"])
        assert len(calls) == 2


@pytest.mark.parametrize("status", [429, 503])
def test_retry_transient_error(status, monkeypatch):
    monkeypatch.setattr("df_jev.client.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(status, headers={"retry-after": "0"})
            if len(calls) == 1
            else httpx.Response(200, json=response())
        )

    with make_client(handler) as client:
        result = JevFrame(pd.DataFrame({"message": ["a"]}), client).evaluate(
            "Condition", columns=["message"]
        )
    assert len(calls) == 2
    assert result.match_error.isna().all()


def test_authentication_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401)

    with make_client(handler) as client, pytest.raises(JevError, match="401"):
        client.decide({"message": "a"}, {"type": "noul", "instructions": "Condition"})
    assert len(calls) == 1


def test_empty_frame_and_output_collision():
    with make_client(lambda _: pytest.fail("No request expected")) as client:
        frame = JevFrame(pd.DataFrame({"message": pd.Series(dtype=str)}), client)
        assert frame.filter("Condition", columns=["message"]).empty
        with pytest.raises(ValueError, match="already exist"):
            JevFrame(
                pd.DataFrame({"message": ["a"], "match_error": ["original"]}), client
            ).evaluate("Condition", columns=["message"])


def test_choice_and_score_return_typed_outputs_and_can_be_chained():
    def handler(request):
        question = json.loads(request.content)["questions"]["result"]
        if question["type"] == "choice":
            answer = {
                "type": "choice",
                "choice": "access",
                "confidence": 0.8,
                "probabilities": {"access": 0.9, "other": 0.1},
            }
        else:
            answer = {
                "type": "score",
                "score": 0.8,
                "confidence": 0.4,
                "probabilities": {"0": 0.2, "1": 0.8},
            }
        return httpx.Response(200, json={"model": "test-1", "answers": {"result": answer}})

    with make_client(handler) as client:
        classified = JevFrame(pd.DataFrame({"message": ["cannot log in"]}), client).classify(
            "Topic", ["access", "other"], columns=["message"]
        )
        scored = JevFrame(classified, client).score("Impact", ["Low", "High"], columns=["message"])
    assert scored.category_label.tolist() == ["access"]
    assert scored.score_value.tolist() == [0.8]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.9", -0.1])
def test_reject_malformed_probability(value):
    with (
        make_client(lambda _: httpx.Response(200, content=json.dumps(response(value)))) as client,
        pytest.raises(JevError, match="invalid decision"),
    ):
        client.decide({"message": "a"}, {"type": "noul", "instructions": "Condition"})


def test_v1_base_url_is_not_duplicated():
    def handler(request):
        assert str(request.url) == "http://test/v1/systemone"
        return httpx.Response(200, json=response())

    with JevClient("http://test/v1/", transport=httpx.MockTransport(handler)) as client:
        client.decide({"message": "a"}, {"type": "noul", "instructions": "Condition"})
