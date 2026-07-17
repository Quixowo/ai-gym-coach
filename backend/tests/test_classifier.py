"""Injury red-flag classifier tests — fully mocked, no live API.

Patches ``get_anthropic_client`` at the classifier's import boundary with a fake
async client whose ``messages.create`` returns a canned single-word response (or
raises). Verifies the strict one-word check and the fail-open-to-ROUTINE behavior:
only exact ``ACUTE_INJURY`` -> True; ROUTINE / unexpected text / whitespace / an API
exception -> False.
"""

from __future__ import annotations

import pytest

import app.agent.classifier as classifier_module
from app.agent.classifier import classify_acute_injury


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, text: str | None, raises: Exception | None) -> None:
        self._text = text
        self._raises = raises

    async def create(self, **kwargs):
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str | None = None, raises: Exception | None = None) -> None:
        self.messages = _FakeMessages(text, raises)


def _patch_client(
    monkeypatch: pytest.MonkeyPatch, text: str | None = None, raises: Exception | None = None
) -> None:
    monkeypatch.setattr(
        classifier_module, "get_anthropic_client", lambda: _FakeClient(text, raises)
    )


async def test_acute_injury_label_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, text="ACUTE_INJURY")
    assert await classify_acute_injury("my knee popped and now it won't bend") is True


async def test_routine_label_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, text="ROUTINE")
    assert await classify_acute_injury("legs are sore after squats") is False


async def test_label_with_surrounding_whitespace_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, text="  ACUTE_INJURY\n")
    assert await classify_acute_injury("sharp pain in my shoulder") is True


async def test_unexpected_text_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, text="I think this might be an injury, you should...")
    assert await classify_acute_injury("hmm") is False


async def test_empty_response_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, text="")
    assert await classify_acute_injury("hmm") is False


async def test_api_exception_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, raises=RuntimeError("API down"))
    assert await classify_acute_injury("my back is on fire") is False
