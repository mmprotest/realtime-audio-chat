from __future__ import annotations

import os
import pytest

pytest.importorskip("gradio")

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from src.app import _chatbot_to_messages, _normalize_chat_history


def test_chatbot_to_messages_dict_entries():
    chatbot = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert _chatbot_to_messages(chatbot) == chatbot


def test_chatbot_to_messages_tuple_entries():
    chatbot = [("hi", "there"), ("again", None)]
    assert _chatbot_to_messages(chatbot) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
        {"role": "user", "content": "again"},
    ]


def test_chatbot_to_messages_string_entries():
    chatbot = ["hola", "bonjour"]
    assert _chatbot_to_messages(chatbot) == [
        {"role": "user", "content": "hola"},
        {"role": "user", "content": "bonjour"},
    ]


@pytest.mark.parametrize(
    "input_history,expected",
    [
        (None, []),
        ({"role": "user", "content": "hi"}, [{"role": "user", "content": "hi"}]),
        ("hello", [{"role": "user", "content": "hello"}]),
        ([("hi", "there")], [("hi", "there")]),
        ([{"role": "assistant", "content": "hey"}], [{"role": "assistant", "content": "hey"}]),
    ],
)
def test_normalize_chat_history(input_history, expected):
    assert _normalize_chat_history(input_history) == expected
