import os
from typing import Literal, Union

from openai import OpenAI


class OpenAIClient:
    def __init__(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]):
        self.client: OpenAI = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def embedding(self, text: Union[str, list], **kwargs):
        if self.model not in ["text-embedding-3-small", "text-embedding-3-large"]:
            raise ValueError(f'not supported model for embedding request -> {self.model}')
        _parsing = kwargs.pop('parsing') if 'parsing' in kwargs else True
        if isinstance(text, str):
            text = [text]
        response = self.client.embeddings.create(input=text, model=self.model, **kwargs)
        if _parsing:
            return response.data[0].embedding
        return response
