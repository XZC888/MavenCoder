from sklearn.cluster import KMeans
import numpy as np
from typing import *

def compute_token_confidence(tokens, strategy = "entropy") -> List[float]:  # mean_prob

    def is_template_token(token_text: str) -> bool:
        if not token_text.strip():
            return True
        return any(p in token_text for p in ["\n", "```", "**", "#", "-"])

    def first_valid_token(tokens):
        for idx, token in enumerate(tokens):
            if "\n" in token.token:
                return idx

    confs = []
    first_valid_idx = first_valid_token(tokens)

    for idx, token in enumerate(tokens):
        if idx < first_valid_idx:
            confs.append(float("inf"))
            continue

        top_k_tokens = [log_prob.token for log_prob in token.top_logprobs]

        if is_template_token(token.token) or any(is_template_token(t) for t in top_k_tokens):
            confs.append(float("inf"))
            continue
        
        if strategy == "mean_prob":
            # mean logprob based estimation
            top_k_logprobs = [log_prob.logprob for log_prob in token.top_logprobs]
            mean_logprob = np.mean(top_k_logprobs)
            confs.append(round(-mean_logprob, 3))

        elif strategy == "entropy":
            # entropy based estimation
            top_k_logprobs = [log_prob.logprob for log_prob in token.top_logprobs]
            ps = np.array([np.exp(log_p) for log_p in top_k_logprobs])
            logps = np.array([log_p for log_p in top_k_logprobs])
            confs.append(round(np.sum(ps * logps), 3))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return confs


def compute_inference_confidence(tokens, confs: List[float], group_size: int = 5) -> List[Tuple[float, List[str]]]:
    sliding_means = []

    for i in range(len(confs) - group_size + 1):
        window_confs = confs[i:i + group_size]
        window_tokens = [tokens[j].token for j in range(i, i + group_size)]

        if float("inf") in window_confs:
            continue

        mean_conf = round(sum(window_confs) / len(window_confs), 3)
        sliding_means.append((mean_conf, window_tokens))

    return sliding_means


def adaptive_predict(inference_confs, theta_1, theta_2):

    conf_values = [x[0] for x in inference_confs]

    def kmeans_threshold(arr, k=2):
        arr = np.array(arr).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(arr)
        centers = sorted(kmeans.cluster_centers_.flatten())
        return centers[0], centers[1]
    
    t1, t2 = kmeans_threshold(conf_values)
    n = len(conf_values)

    l = sum(1 for c in conf_values if c <= t1) / n
    h = sum(1 for c in conf_values if c >= t2) / n

    return "easy" if h >= theta_1 else("hard" if l >= theta_2 else "medium")

