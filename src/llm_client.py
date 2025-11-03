from typing import Optional


class LLMClient:
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        raise NotImplementedError


class EchoLLM(LLMClient):
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        # 占位：截断回显，保证流程可跑
        return prompt[-800:]


class ApiLLMClient(LLMClient):
    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        # 预留：接入你的实际 API（OpenAI/本地推理服务/其他）
        # 例如：requests.post(base_url, json={"model": self.model, "messages": [{"role":"user","content":prompt}], ...})
        raise NotImplementedError("Connect your real API here.")
