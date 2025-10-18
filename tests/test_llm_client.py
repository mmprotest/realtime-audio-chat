import sys
import types

from src import llm_client


def _make_chat_response(text: str):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def test_llm_client_retries_without_proxies(monkeypatch):
    init_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            if len(init_calls) == 1:
                raise TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")
            self.kwargs = kwargs
            completions = types.SimpleNamespace(create=lambda **_: _make_chat_response("ok"))
            self.chat = types.SimpleNamespace(completions=completions)

    fake_http_client_instances = []

    class FakeHttpClient:
        def __init__(self, trust_env: bool = True):
            self.trust_env = trust_env
            fake_http_client_instances.append(self)

        def close(self):
            pass

    fake_httpx_module = types.SimpleNamespace(Client=FakeHttpClient)

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx_module)

    client = llm_client.LLMClient(api_key="key", base_url=None)

    assert len(init_calls) == 2
    assert "http_client" in init_calls[1]
    assert fake_http_client_instances[0].trust_env is False

    result = client.chat([{"role": "user", "content": "hi"}], model="test")
    assert result == "ok"

    client.close()

