import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.encode_decode import encode, decode

import tiktoken

output_folder = os.path.join(os.getcwd(), "output_files")

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_vocab_merges_regex(tokenizer):
    
    tokenizer_path = f"tokenizer_{tokenizer}"
    tokenizer_folder = os.path.join(output_folder, tokenizer_path)
    
    for file in os.listdir(tokenizer_folder):
        
        if "vocab" in file:
            with open(os.path.join(tokenizer_folder, file), "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                
        if "merges" in file:
            with open(os.path.join(tokenizer_folder, file), "r", encoding="utf-8") as f:
                merges = {}
                
                for i, line in enumerate(f):
                    p0, p1 = map(int, line.split())
                    merges[(p0, p1)] = (256 + i)
                    
        if "config" in file:
            with open(os.path.join(tokenizer_folder, file), "r") as f:
                config_data = json.load(f)
                
    vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
    regex_pattern = config_data["regex_pattern"]
    
    return vocab, merges, regex_pattern

def encode_decode(text, vocab_size, tokenizer):
    
    vocab, merges, regex_pattern = get_vocab_merges_regex(tokenizer)
    
    tokens = encode(text, vocab, merges, regex_pattern, vocab_size)
    
    decoded_tokens = []
    for token in tokens:
        decoded_tokens.append(decode([token], vocab, vocab_size))
        
    decoded_text = decode(tokens, vocab, vocab_size)
        
    # gpt2 and gpt4 encoding
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    tokens_gpt2 = enc_gpt2.encode(text, allowed_special={"<|endoftext|>"})
    decoded_tokens_gpt2 = []
    
    for token in tokens_gpt2:
        token_bytes = enc_gpt2.decode_single_token_bytes(token)
        decoded_tokens_gpt2.append(token_bytes.decode("utf-8", errors="replace"))
        
    enc_gpt4 = tiktoken.get_encoding("cl100k_base")
    tokens_gpt4 = enc_gpt4.encode(text, allowed_special={"<|endoftext|>"})
    decoded_tokens_gpt4 = []
    
    for token in tokens_gpt4:
        token_bytes = enc_gpt4.decode_single_token_bytes(token)
        decoded_tokens_gpt4.append(token_bytes.decode("utf-8", errors="replace"))
    
    return {
        "tokens_custom": tokens,
        "decoded_tokens_custom": decoded_tokens,
        "match": (decoded_text == text),
        "tokens_gpt2": tokens_gpt2,
        "decoded_tokens_gpt2": decoded_tokens_gpt2,
        "tokens_gpt4": tokens_gpt4,
        "decoded_tokens_gpt4": decoded_tokens_gpt4,
        "vocab_size": vocab_size,
        "vocab_size_gpt2": enc_gpt2.n_vocab,
        "vocab_size_gpt4": enc_gpt4.n_vocab
    }