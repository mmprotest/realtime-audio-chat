from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from src.app import _chatbot_to_messages


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
