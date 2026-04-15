from typing import List, Union, Optional, Literal
import dataclasses
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import random
import  time
import openai
import numpy as np
from .trans_format import dict_to_chat_token

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


def change_messages(messages, max_len):
    encoder = tiktoken.encoding_for_model("gpt-4")
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(encoder.encode(l))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(encoder.encode(messages[0].content))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(encoder.encode(msg.content))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages


def change_messages_for_local_model(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.tokenize(l))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.tokenize(messages[0].content))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.tokenize(msg.content))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages



class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 4096, stop_strs: Optional[List[str]] = None, temperature: float = 0.2, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(
        self,
        model_name: str,
        key: str = "",
        url: str = "",
        embedding_key: str = "",
        embedding_url: str = "",
        embedding_model: str = "text-embedding-3-large",
    ):
        self.name = model_name
        self.is_chat = True
        if not key:
            raise ValueError("OpenAI KEY are required!")

        kwargs = {"api_key": key}
        if url:
            kwargs["base_url"] = url

        self.client = openai.OpenAI(**kwargs)
        self.embedding_model = embedding_model
        self.request_extra_body = None
        
        embedding_kwargs = {"api_key": embedding_key if embedding_key else key}
        if embedding_url:
            embedding_kwargs["base_url"] = embedding_url
        elif url and not embedding_key:
            embedding_kwargs["base_url"] = url
        self.embedding_client = openai.OpenAI(**embedding_kwargs)


    def gpt_chat(
        self,
        messages,
        stop: List[str] = None,
        max_tokens: int = 10000,
        temperature: float = 0.2,
        top_p = 0.8,
        num_comps=1,
    ) -> Union[List[str], str]:
        try:
            new_messages = change_messages(messages, max_tokens)
            messages = new_messages
            request_kwargs = {
                "model": self.name,
                "messages": [dataclasses.asdict(message) for message in messages],
                "temperature": temperature,
                "top_p": top_p,
                "n": num_comps,
                "stop": stop,
                "timeout": 120,
            }
            if self.request_extra_body is not None:
                request_kwargs["extra_body"] = self.request_extra_body
            response = self.client.chat.completions.create(
                **request_kwargs,
            )
        except Exception as e:
            print("GPT Error:", str(e))
            raise RuntimeError("GPT API error: ", str(e))
        if num_comps == 1:
            return response.choices[0].message.content  # type: ignore
        return [choice.message.content for choice in response.choices]  # type: ignore

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = 7500, temperature: float = 0.0, num_comps: int = 1, top_p: float = 1.0) -> Union[List[str], str]:
        res = self.gpt_chat(messages, stop, max_tokens, temperature, top_p, num_comps)
        return res

    def get_embedding(self, text):
        data = self.embedding_client.embeddings.create(model=self.embedding_model, input=text).data
        
        if isinstance(text, list):
            return [np.array(e.embedding) for e in data]
        return np.array(data[0].embedding)
    
    def generate_tokens(self, messages: List[Message], k: int = 5, max_tokens: int = 7500, temperature: float = 0.0, top_p: float = 1.0):
        new_messages = change_messages(messages, max_tokens)
        messages = new_messages
        request_kwargs = {
            "model": self.name,
            "messages": [dataclasses.asdict(message) for message in messages],
            "logprobs": True,
            "top_logprobs": k,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.request_extra_body is not None:
            request_kwargs["extra_body"] = self.request_extra_body
        response = self.client.chat.completions.create(**request_kwargs)
        
        if response.choices[0].logprobs:
            tokens = response.choices[0].logprobs.content
        else:
            tokens = [dict_to_chat_token(tok) for tok in response.choices[0].message.logprobs['content']]

        response = response.choices[0].message.content

        return tokens, response

class GPT4o_mini(GPTChat):
    def __init__(self, key, url, embedding_key="", embedding_url="", embedding_model: str = "text-embedding-3-large"):
        super().__init__("gpt-4o-mini", key, url, embedding_key, embedding_url, embedding_model)

class GPT4_1_nano(GPTChat):
    def __init__(self, key, url, embedding_key="", embedding_url="", embedding_model: str = "text-embedding-3-large"):
        super().__init__("gpt-4.1-nano", key, url, embedding_key, embedding_url, embedding_model)

class O1_MINI(GPTChat):
    def __init__(self, key, url, embedding_key="", embedding_url="", embedding_model: str = "text-embedding-3-large"):
        super().__init__("o1-mini", key, url, embedding_key, embedding_url, embedding_model)

class QWEN(GPTChat):
    def __init__(self, key, url, embedding_key="", embedding_url="", embedding_model: str = "text-embedding-3-large"):
        super().__init__("qwen3-coder-plus", key, url, embedding_key, embedding_url, embedding_model)

class GLM_4_7(GPTChat):
    def __init__(self, key, url, embedding_key="", embedding_url="", embedding_model: str = "text-embedding-3-large"):
        super().__init__("glm-4.7", key, url, embedding_key, embedding_url, embedding_model)
        self.request_extra_body = {"thinking": {"type": "disabled"}}


class VLLMModelBase(ModelBase):
    """
    Huggingface chat models
    """

    def __init__(self, model, url=""):
        super().__init__(model)
        self.model = model
        self.is_chat = True
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=url if url else f"http://localhost:8000/v1")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 8000

    def vllm_chat(
            self,
            prompt: str,
            stop: List[str] = [""],
            max_tokens: int = 8192,
            temperature: float = 0.0,
            num_comps=1,
    ) -> Union[List[str], str]:
        max_length = self.max_length
        Internal_Server_Error = 0
        request_timeout = 0
        timeout = 1800
        while True:
            prompt = change_messages_for_local_model(self.tokenizer, prompt, max_length)  # StarCoder max length
            try:
                responses = self.vllm_client.chat.completions.create(
                    model=self.model,
                    messages=[dataclasses.asdict(message) for message in prompt],
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    # stop=stop,
                    # frequency_penalty=0.0,
                    # presence_penalty=0.0,
                    n=num_comps,
                    timeout=timeout,
                )
            except Exception as e:
                print("VLLM Error:", str(e))
                if "Internal Server Error" in str(e):
                    if Internal_Server_Error <= 5:
                        print("try again Server Error")
                        num = round(random.uniform(0, 2), 4)
                        time.sleep(num + Internal_Server_Error)
                        Internal_Server_Error += 1
                        continue

                    else:
                        print("try 5 times Internal Server Error")
                        assert False, "VLLM API error: " + str(e)
                if "Request timed out" in str(e):
                    if request_timeout <= 5:
                        max_tokens = 8192 - 1024
                        print("try again Request timed out")
                        num = round(random.uniform(0, 2), 4)
                        time.sleep(num + request_timeout)
                        request_timeout += 1
                        continue
                    else:
                        print("try 5 Request timed out")
                        assert False, "VLLM API error: " + str(e)
                if "maximum context length" in str(e):
                    max_length -= 2000
                else:
                    assert False, "VLLM API error: " + str(e)
            else:
                break
        if num_comps == 1:
            return responses.choices[0].message.content  # type: ignore
        return [responses.choices[0].message.content for response in responses]  # type: ignore

    def generate_completion(self, messages: str, stop: List[str] = [""], max_tokens: int = 8192,
                            temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        ret = self.vllm_chat(messages, stop, max_tokens, temperature, num_comps)
        return ret

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = 8192,
                      temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        res = self.generate_completion(messages, stop, max_tokens, temperature, num_comps)
        return res

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += message.content + "\n"
            if i == len(messages) - 1:
                prompt += "\n"
        return prompt

    def extract_output(self, output: str) -> str:
        return output



class DeepSeek(VLLMModelBase):
    def __init__(self, url=""):
        super().__init__("phi-3", url)


class Llama(VLLMModelBase):
    def __init__(self,  url=""):
        super().__init__("Llama3.0", url)
