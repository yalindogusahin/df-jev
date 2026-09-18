import json

import httpx
import pandas as pd
import pytest

from jevpandas import (
    JevClient,
    JevError,
    JevFrame,
    JevResult,
    choice,
    noul,
    score,
    tqdm_progress,
)


def response(request, value=0.9):
    names = list(json.loads(request.content)["questions"])
    return {
        "model": "test-1",
        "answers": {name: {"type": "noul", "noul": value} for name in names},
    }


def make_client(handler, **kwargs):
    return JevClient(
        "http://test", "dummy", "test-1", transport=httpx.MockTransport(handler), **kwargs
    )


def test_duplicate_indices_missing_values_and_selected_fields():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200, json=response(request, 0.9 if payload["state"]["message"] else 0.1)
        )

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
        return httpx.Response(200, json=response(request))

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
        return httpx.Response(200, json=response(request, 1.8))

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
    monkeypatch.setattr("jevpandas.client.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(status, headers={"retry-after": "0"})
            if len(calls) == 1
            else httpx.Response(200, json=response(request))
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
        qname, question = next(iter(json.loads(request.content)["questions"].items()))
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
        return httpx.Response(200, json={"model": "test-1", "answers": {qname: answer}})

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
        make_client(
            lambda r: httpx.Response(200, content=json.dumps(response(r, value)))
        ) as client,
        pytest.raises(JevError, match="invalid decision"),
    ):
        client.decide({"message": "a"}, {"type": "noul", "instructions": "Condition"})


def test_v1_base_url_is_not_duplicated():
    def handler(request):
        assert str(request.url) == "http://test/v1/systemone"
        return httpx.Response(200, json=response(request))

    with JevClient("http://test/v1/", transport=httpx.MockTransport(handler)) as client:
        client.decide({"message": "a"}, {"type": "noul", "instructions": "Condition"})


def test_parallel_run_preserves_order_and_isolates_failed_rows():
    def handler(request):
        message = json.loads(request.content)["state"]["message"]
        if message == "bad":
            return httpx.Response(200, json=response(request, 1.8))
        return httpx.Response(200, json=response(request, {"a": 0.9, "b": 0.5, "c": 0.1}[message]))

    with make_client(handler) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a", "b", "c", "a", "b", "bad"]}), client)
        result = frame.evaluate("Condition", columns=["message"], workers=4)
    assert result.match_probability.iloc[:5].tolist() == [0.9, 0.5, 0.1, 0.9, 0.5]
    assert result.match_probability.isna().iloc[-1]
    assert result.match_error.isna().tolist() == [True, True, True, True, True, False]
    assert result.index.tolist() == list(range(6))


def test_parallel_cache_is_thread_safe():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(request))

    with make_client(handler) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a"] * 50}), client)
        result = frame.evaluate("Condition", columns=["message"], workers=8)
    assert result.match_error.isna().all()
    assert result.match_probability.notna().all()
    assert result.match_cached.sum() >= 1
    assert 0 < len(calls) <= 8  # one request per distinct row, deduplicated under concurrency


def test_invalid_workers_is_rejected():
    with make_client(lambda _: pytest.fail("No request expected")) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a"]}), client)
        with pytest.raises(ValueError, match="Workers"):
            frame.evaluate("Condition", columns=["message"], workers=0)
        with pytest.raises(ValueError, match="Workers"):
            frame.evaluate("Condition", columns=["message"], workers=1.5)


def test_ask_sends_all_questions_in_one_request_per_row():
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9}
            elif question["type"] == "choice":
                answers[name] = {
                    "type": "choice",
                    "choice": "access",
                    "confidence": 0.8,
                    "probabilities": {label: 0.5 for label in question["criteria"]},
                }
            else:
                answers[name] = {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.7,
                    "probabilities": {str(i): 0.5 for i in range(len(question["criteria"]))},
                }
        return httpx.Response(200, json={"model": "test-1", "answers": answers})

    with make_client(handler) as client:
        frame = JevFrame(pd.DataFrame({"message": ["cannot log in", "refund please"]}), client)
        result = frame.ask(
            {
                "access": noul("Is this an access problem?"),
                "topic": choice("Topic?", {"access": "Login", "billing": "Payments"}),
                "severity": score("Severity?", ["Low", "High"]),
            },
            columns=["message"],
        )
    assert len(payloads) == 2  # one request per row, not one per question
    assert set(payloads[0]["questions"]) == {"access", "topic", "severity"}
    assert result.access_probability.tolist() == [0.9, 0.9]
    assert result.topic_label.tolist() == ["access", "access"]
    assert result.severity_value.tolist() == [1.0, 1.0]
    assert result.access_error.isna().all()


def test_ask_validates_questions_and_names():
    with make_client(lambda _: pytest.fail("No request expected")) as client:
        frame = JevFrame(pd.DataFrame({"message": ["a"]}), client)
        with pytest.raises(ValueError, match="at least one"):
            frame.ask({}, columns=["message"])
        with pytest.raises(TypeError, match="mapping"):
            frame.ask([noul("X")], columns=["message"])
        with pytest.raises(ValueError, match="nonempty"):
            frame.ask({"": noul("X")}, columns=["message"])
        with pytest.raises(ValueError, match="Unknown question"):
            frame.ask({"q": {"type": "bogus"}}, columns=["message"])


def test_result_is_jevresult_with_metadata_and_repr():
    with make_client(lambda r: httpx.Response(200, json=response(r))) as client:
        result = JevFrame(pd.DataFrame({"message": ["a", "a"]}), client).evaluate(
            "Condition", columns=["message"]
        )
    assert isinstance(result, JevResult)
    assert result.metadata["rows"] == 2
    assert result.metadata["cache_hits"] == 1
    html = result._repr_html_()
    assert "jev" in html and "rows" in html


def test_tqdm_progress_adapter_reports_cumulative_counts():
    class FakeBar:
        def __init__(self):
            self.total = None
            self.n = 0
            self.updates = []

        def update(self, n):
            self.n += n
            self.updates.append(self.n)

    bar = FakeBar()
    report = tqdm_progress(bar)
    report(2, 10)
    report(5, 10)
    assert bar.total == 10
    assert bar.updates == [2, 5]
