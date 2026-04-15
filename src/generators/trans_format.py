from dataclasses import dataclass
from typing import *


@dataclass
class TopLogprob:
    token: str
    bytes: List[int]
    logprob: float

@dataclass
class ChatCompletionTokenLogprob:
    token: str
    bytes: List[int]
    logprob: float
    top_logprobs: List[TopLogprob]


def dict_to_chat_token(d) -> ChatCompletionTokenLogprob:
    top_list = []
    for tl in d["top_logprobs"]:
        top_list.append(TopLogprob(
            token=tl["token"],
            bytes=tl["bytes"],
            logprob=tl["logprob"]
        ))

    return ChatCompletionTokenLogprob(
        token=d["token"],
        bytes=d["bytes"],
        logprob=d["logprob"],
        top_logprobs=top_list
    )