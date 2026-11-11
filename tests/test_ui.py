from streamlit.testing.v1 import AppTest


def test_blank_or_one_character_question_is_not_sent(monkeypatch):
    import requests
    sent = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(k) or (_ for _ in ()).throw(
        requests.ConnectionError("the test must not reach the API")))
    for typed, warns in (("   ", False), ("a", True), (" ক ", True)):
        at = AppTest.from_file("../ui/app.py", default_timeout=30).run()
        at.chat_input[0].set_value(typed).run()
        assert not at.exception
        assert sent == [] and at.session_state.messages == []
        assert bool(at.warning) == warns  # blank input does nothing; one character gets a hint
