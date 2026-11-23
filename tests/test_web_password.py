import pytest

from godoc_agent.web import password


def _feed(monkeypatch, *answers):
    replies = iter(answers)
    monkeypatch.setattr(password.getpass, "getpass", lambda prompt="": next(replies))


def test_rejects_password_shorter_than_nine_chars(monkeypatch, capsys):
    _feed(monkeypatch, "a" * 8)
    assert password.main() == 2
    assert "9 文字以上" in capsys.readouterr().err


def test_accepts_nine_char_password(monkeypatch, capsys):
    _feed(monkeypatch, "a" * 9, "a" * 9)
    assert password.main() == 0
    assert capsys.readouterr().out.startswith("WEB_ADMIN_PASSWORD_HASH=")


@pytest.mark.parametrize("second", ["a" * 9 + "x", "b" * 9])
def test_rejects_mismatched_confirmation(monkeypatch, capsys, second):
    _feed(monkeypatch, "a" * 9, second)
    assert password.main() == 2
    assert "一致しません" in capsys.readouterr().err
