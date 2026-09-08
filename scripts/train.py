import regex as re
import json
import os

class JashTokenizer:
    
    def __init__(self, regex_pattern, regex_name, final_vocab_size=50257, path=""):
        self.regex_pattern = regex_pattern
        self.compiled_pattern = re.compile(self.regex_pattern)
        
        self.final_vocab_size = final_vocab_size
        self.folder_path = path
        
        self.special_characters = {
            "<|endoftext|>": final_vocab_size - 1
        }
        
        self.vocab = None
        self.merges = None
        
        self.regex_name = regex_name
        
    def get_stats(self, chunks):
        counts = {}
        
        for chunk in chunks:
            for pair in zip(chunk, chunk[1:]):
                counts[pair] = counts.get(pair, 0) + 1
                
        return counts
    
    def merge(self, ids, pair, idx):
        
        new_ids = []
        
        i = 0
        
        while i < len(ids):
            
            if (i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]):
                new_ids.append(idx)
                i += 2
                
            else:
                new_ids.append(ids[i])
                i += 1
                
        return new_ids
    
    def train(self, text):
        
        self.vocab = {
            idx: bytes([idx]) for idx in range(256)
        }
        
        self.merges = {}
        
        chunks = self.compiled_pattern.findall(text)
        chunks = [list(chunk.encode("utf-8")) for chunk in chunks]
        
        num_merges = (self.final_vocab_size - 256 - len(self.special_characters))
        
        for i in range(num_merges):
            
            stats = self.get_stats(chunks)
            
            if not stats:
                break
            
            max_pair = max(stats, key=stats.get)
            idx = (256 + i)
            
            print(f"Merging {max_pair} into new token {idx}")
            
            new_chunk_list = []
            
            for chunk in chunks:
                
                new_chunk = self.merge(chunk, max_pair, idx)
                new_chunk_list.append(new_chunk)
                
            self.merges[max_pair] = idx
            chunks = new_chunk_list
            
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        for token, idx in self.special_characters.items():
            self.vocab[idx] = token.encode("utf-8")
            
    def save(self):
        
        print(f"Final Vocab example: ")
        print("="*10)
        
        i = 0
        while (i < 10):
            for token, idx in self.vocab.items():
                print(f"{token}: {idx}")
                i += 1
                
        print(".")
        print(".")
        print(".")
        print(f"{self.vocab[self.final_vocab_size - 1]}: {self.final_vocab_size - 1}")
        print("="*10)
        
        print(f"Final Merges example: ")
        print("="*10)
        
        i = 0
        while (i < 10):
            for pair, idx in self.merges.items():
                print(f"{pair}: {idx}")
                i += 1
                
        print(".")
        print(".")
        print(".")
        
        print("="*10)
        
        
        vocab_data = {
            str(idx): list(token_bytes) for idx, token_bytes in self.vocab.items()
        }
        
        with open(os.path.join(self.folder_path, f"vocab_{self.final_vocab_size}_{self.regex_name}.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)
            
        merges_by_id = sorted(self.merges.items(), key=lambda x: x[1])
        
        with open(os.path.join(self.folder_path, f"merges_{self.final_vocab_size}_{self.regex_name}.bpe"), "w", encoding="utf-8") as f:
            
            for (p0, p1), idx in merges_by_id:
                f.write(f"{p0} {p1}\n")
                
        config = {
            "vocab_size": self.final_vocab_size,
            "base_vocab_size": 256,
            "num_merges": len(self.merges),
            "special_tokens": self.special_characters,
            "regex_pattern": self.regex_pattern,
            "regex_name": self.regex_name
        }            
        with open(os.path.join(self.folder_path, "config.json"), "w", encoding="utf-8") as f:
            
            json.dump(config, f, indent=4, ensure_ascii=False)
             
        print("Tokenizer saved. vocab.json, merges.bpe, and config.json created.")
        print("-"*50)