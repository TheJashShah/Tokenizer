import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import warnings

from encode_decode import encode, decode

def calculate_metrics(text, vocab, merges, regex_pattern, vocab_sizes):
    
    results = []
    
    for vocab_size in vocab_sizes:
        encoded = encode(text, vocab, merges, regex_pattern, vocab_size)
        decoded = decode(encoded, vocab, vocab_size)
        
        token_count = len(encoded)
        byte_count = len(text.encode("utf-8"))
        
        results.append({
            "vocab_size": vocab_size,
            "token_count": token_count,
            "unique_tokens": len(set(encoded)),
            "bytes_per_token": byte_count / token_count,
            "tokens_per_count": token_count / byte_count,
            "decode_correct": decoded == text
        })
        
    return pd.DataFrame(results)

def plot_metrics(results, path):
    
    warnings.filterwarnings("ignore")
    
    metrics = [
        "token_count",
        "unique_tokens",
        "bytes_per_token",
        "tokens_per_count"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for ax, metric in zip(axes.flat, metrics):
        
        sns.lineplot(data=results, x="vocab_size", y=metric, marker="o", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Vocabulary Size")
        ax.set_ylabel(metric.replace("_", " ").title())
        
    plt.tight_layout()
    plt.savefig(path) 
    plt.show()
    return plt
    