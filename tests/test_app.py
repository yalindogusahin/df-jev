from pathlib import Path

from streamlit.testing.v1 import AppTest

from df_jev import JevClient

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_preview_threshold_and_stale_results(monkeypatch):
    calls = []

    def decide(self, state, question):
        calls.append((state, question))
        return {"model": "test", "answers": {"result": {"type": "noul", "noul": 0.8}}}, False

    monkeypatch.setattr(JevClient, "decide", decide)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert not calls  # Loading the app never sends data.
    next(b for b in app.button if b.label == "Preview first 10 rows").click().run()
    assert not app.exception
    assert len(calls) == 10
    assert len(app.session_state["run"]["result"]) == 10
    app.slider[0].set_value(0.9).run()
    assert not app.exception
    assert len(calls) == 10
    assert any("0 rows in this view" in text.value for text in app.markdown)
    app.text_area[0].set_value("A different condition").run()
    assert not app.exception
    assert any("changed" in text.value for text in app.info)
    assert len(calls) == 10


def test_classification_and_scoring_views(monkeypatch):
    def decide(self, state, question):
        if question["type"] == "choice":
            answer = {
                "type": "choice",
                "choice": "access",
                "confidence": 0.8,
                "probabilities": {"access": 0.8, "billing": 0.1, "technical": 0.05, "other": 0.05},
            }
        else:
            answer = {
                "type": "score",
                "score": 1.5,
                "confidence": 0.7,
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            }
        return {"model": "test", "answers": {"result": answer}}, False

    monkeypatch.setattr(JevClient, "decide", decide)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    next(r for r in app.radio if r.label == "Operation").set_value("Classify").run()
    next(b for b in app.button if b.label == "Preview first 10 rows").click().run()
    assert not app.exception
    assert app.session_state["run"]["result"].jev_label.eq("access").all()
    next(r for r in app.radio if r.label == "Operation").set_value("Score").run()
    next(b for b in app.button if b.label == "Preview first 10 rows").click().run()
    assert not app.exception
    assert app.session_state["run"]["result"].jev_value.eq(1.5).all()


def test_invalid_endpoint_does_not_close_the_working_client():
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    client = app.session_state["client"]
    endpoint = next(w for w in app.text_input if w.label == "Endpoint")
    original = endpoint.value
    endpoint.set_value("not-a-url").run()
    assert not app.exception
    assert app.error
    assert not client._http.is_closed
    next(w for w in app.text_input if w.label == "Endpoint").set_value(original).run()
    assert not app.exception
    assert not app.session_state["client"]._http.is_closed
