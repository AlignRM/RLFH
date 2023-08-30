import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import openai

logger = logging.getLogger(__name__)


@dataclass
class CompletionConfig:
    model: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "Openai interface args."
        },
    )

    model_url: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )

    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )
    api_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )
    api_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )
    api_base: Optional[str] = field(
        default=None,
        metadata={
            "help": "Openai interface args."
        },
    )


class Completion:
    def __init__(self, config: CompletionConfig):
        self.model = config.model
        self.config = config
        self.system_prompt = config.system_prompt
        self.client = openai.OpenAI(
            # Replace the URL if deploying your app remotely
            # (e.g., on Anyscale or KubeRay).
            base_url=config.model_url,
            api_key="NOT A REAL KEY",
        )

    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def complete(
        self,
        message: str,
        temperature: Optional[float] = 1,
        top_p: Optional[float] = 1,
        max_tokens: Optional[int] = None,
        sleep_time: int = 1,
    ):
        while True:
            try:
                messages = [{"role": "user", "content": message}]
                if self.system_prompt is not None:
                    messages = [{"role": "system", "content": self.system_prompt}] + message
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
                choice = completion.choices[0]
                return choice.message.content
            except Exception as e:
                logger.warning(f"Error: {e}.")
                logger.info(f"Sleeping {sleep_time} before retrying to call openai API...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
