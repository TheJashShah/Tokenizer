import regex as re

def encode(text, vocab, merges, regex_pattern, vocab_size=50257):
    
    compiled_pattern = re.compile(regex_pattern)
    
    special_token = "<|endoftext|>"
    special_token_id = max(vocab.keys())
    
    parts = text.split(special_token)
    encoded  = []
    
    for i, part in enumerate(parts):
        
        regex_chunks = compiled_pattern.findall(part)
        
        for chunk in regex_chunks:
            chunk = list(chunk.encode("utf-8"))
            
            while True:
                stats = {}
                
                for pair in zip(chunk, chunk[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
                    
                if not stats:
                    break
                
                pair = min(stats, key=lambda p: merges.get(p, float("inf")))
                
                if pair not in merges:
                    break
                
                if merges[pair] >= vocab_size:
                    break
                
                chunk = merge(chunk, pair, merges[pair])
                
            encoded.extend(chunk)
            
        if i < len(parts) - 1:
            encoded.append(special_token_id)
            
    return encoded

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    
    while i < len(ids):
        
        if(i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]):
            new_ids.append(idx)
            i += 2
            
        else:
            new_ids.append(ids[i])
            i += 1
            
    return new_ids

def decode(ids, vocab, vocab_size):
    special_token_id = max(vocab.keys())
    special_token = "<|endoftext|>"
    
    byte_stream = b""
    
    for idx in ids:
        
        if idx == special_token_id:
            byte_stream += special_token.encode("utf-8")
        else:
            byte_stream += vocab[idx]
            
    return byte_stream.decode("utf-8", errors="ignore")