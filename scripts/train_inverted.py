import regex as re
import json
import os

from collections import Counter, defaultdict

class InvertedJashTokenizer:
    
    def __init__(self, regex_pattern, regex_name, final_vocab_size, path):
        self.regex_pattern = regex_pattern
        self.regex_name = regex_name
        self.final_vocab_size = final_vocab_size
        self.folder_path = path
        self.compiled_pattern = re.compile(self.regex_pattern)
        
        self.special_characters = {
            "<|endoftext|>": final_vocab_size - 1
        }
        
        self.vocab = None
        self.merges = None
        
    def get_best_pair(self, pair_counts):
        
        return max(pair_counts, key=lambda p: (pair_counts[p], p))
    
    def build_unique_chunks(self, chunks):
        
        counter = Counter(tuple(chunk) for chunk in chunks)
        
        unique_chunks = []
        frequencies = []
        
        for chunk, frequency in counter.items():
            unique_chunks.append(chunk)
            frequencies.append(frequency)
            
        return unique_chunks, frequencies
    
    def build_inverted_index(self, unique_chunks, frequencies):
        
        pair_counts = Counter()
        pair_to_chunks = defaultdict(set)
        
        for chunk_idx, chunk in enumerate(unique_chunks):
            freq = frequencies[chunk_idx]
            
            for pair in zip(chunk, chunk[1:]):
                pair_counts[pair] += freq
                pair_to_chunks[pair].add(chunk_idx)
                
        return pair_counts, pair_to_chunks
    
    def merge_unique_chunk(self, chunk, pair, idx):
        
        new_chunk = []
        i = 0
        
        while i < len(chunk):
            
            if (i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i + 1] == pair[1]):
                new_chunk.append(idx)
                i += 2
                
            else:
                new_chunk.append(chunk[i])
                i += 1
                
        return new_chunk
    
    def train(self, text):
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        
        chunks = self.compiled_pattern.findall(text)
        chunks = [list(chunk.encode("utf-8")) for chunk in chunks]
        
        unique_chunks, frequencies = self.build_unique_chunks(chunks)
        pair_counts, pair_to_chunks = self.build_inverted_index(unique_chunks, frequencies)
        
        num_merges = (self.final_vocab_size - 256 - len(self.special_characters))
        
        for i in range(num_merges):
            
            if not pair_counts:
                break
            
            max_pair = self.get_best_pair(pair_counts)
            idx =  (256 + i)
            
            if i % 1000 == 0:
                print(f"Merge {i} | Merging {max_pair} into new token {idx}")
            
            affected_chunks = list(pair_to_chunks.get(max_pair, set()))
            
            for chunk_idx in affected_chunks:
                old_chunk = unique_chunks[chunk_idx]
                freq = frequencies[chunk_idx]
                
                old_pairs = Counter(zip(old_chunk, old_chunk[1:]))
                
                for pair, count in old_pairs.items():
                    
                    pair_counts[pair] -= (count * freq)
                    
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                        
                    if pair in pair_to_chunks:
                        pair_to_chunks[pair].discard(chunk_idx)
                        
                        if not pair_to_chunks[pair]:
                            del pair_to_chunks[pair]
                            
                new_chunk = self.merge_unique_chunk(old_chunk, max_pair, idx)
                
                unique_chunks[chunk_idx] = new_chunk
                
                new_pairs = Counter(zip(new_chunk, new_chunk[1:]))
                
                for pair, count in new_pairs.items():
                    pair_counts[pair] += (count * freq)
                    
                    pair_to_chunks[pair].add(chunk_idx)
                    
            self.merges[max_pair] = idx
            
        
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = (self.vocab[p0] + self.vocab[p1])
        
        for token, idx in self.special_characters.items():
            self.vocab[idx] = token.encode("utf-8")
            
    def save(self):
        
        print("Final Vocab example: ")
        print("="*10)
        
        i = 0
        while i < 10:
            
            for token, idx in self.vocab.items():
                
                print(f"{token}: {idx}")
                
                i += 1
                if i >= 10:
                    break
                
        print(".")
        print(".")
        print(".")
        
        print(f"{self.vocab[self.final_vocab_size - 1]}: {self.final_vocab_size - 1}")
        
        print("="*10)
        
        print("Final Merges example: ")
        print("=" * 10)
        
        i = 0
        while i < 10:
            for pair, idx in self.merges.items():
                
                print(f"{pair}: {idx}")
                i += 1
                
                if i >= 10:
                    break
                
        print(".")
        print(".")
        print(".")
        
        print("=" * 10)
        
        vocab_data = {str(idx): list(token_bytes) for idx, token_bytes in self.vocab.items()}
        
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
        
        print("-" * 50)