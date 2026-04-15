from .model import Llama, ModelBase, DeepSeek, GPT4o_mini, GPT4_1_nano, O1_MINI, QWEN, GLM_4_7

def model_factory(
    model_name: str,
    key: str = "",
    url: str = "",
    embedding_key: str = "",
    embedding_url: str = "",
    embedding_model: str = "text-embedding-3-large",
) -> ModelBase:
    if model_name == "gpt-4o-mini":
        return GPT4o_mini(key, url, embedding_key, embedding_url, embedding_model)
    elif model_name == "gpt-4.1-nano":
        return GPT4_1_nano(key, url, embedding_key, embedding_url, embedding_model)
    elif model_name == "o1-mini":
        return O1_MINI(key, url, embedding_key, embedding_url, embedding_model)
    elif model_name == "qwen3-coder-plus":
        return QWEN(key, url, embedding_key, embedding_url, embedding_model)
    elif model_name == "glm-4.7":
        return GLM_4_7(key, url, embedding_key, embedding_url, embedding_model)
    # elif model_name == "llama3":
    #     return Llama(url)
    # elif model_name == "deepseek-coder":
    #     return DeepSeek(url)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
