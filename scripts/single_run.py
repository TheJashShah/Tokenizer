import os
import json

from experiments import calculate_metrics, plot_metrics
from train import JashTokenizer

'''
Run data_collection as a seperate script, then just input the dataset folder, and the output folder name, to train the tokenizer, then run experiments on it.
'''

data_folder = os.path.join(os.getcwd(), "datasets", "trial_run_data_collection") # tokenizer train text
with open(os.path.join(data_folder, "output.txt"), "r", encoding="utf-8") as f:
    train_text = f.read()

folder_path = os.path.join(os.getcwd(), "output_files", "trial_tokenizer")
os.makedirs(folder_path, exist_ok=True)# output folder

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

print("="*5)
print("Training Tokenizer: ")

tokenizerOBJ = JashTokenizer(GPT4_SPLIT_PATTERN, "GPT4", 557, folder_path)
tokenizerOBJ.train(train_text)
tokenizerOBJ.save()

# print("="*5)

for file in os.listdir(folder_path):
    
    if "vocab" in file:
        with open(os.path.join(folder_path, file), "r") as f:
            vocab_data = json.load(f)
    
    if "merges" in file:
        with open(os.path.join(folder_path, file)) as f:
            merges = {}
            
            for i, line in enumerate(f):
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = (256 + i)
                
    if "config" in file:
        with open(os.path.join(folder_path, file), "r") as f:
            config_data = json.load(f)
            
vocab = {
    int(k): bytes(v) for k, v in vocab_data.items()
}
regex_pattern = config_data["regex_pattern"]
print(regex_pattern)
vocab_sizes = [100, 150, 200, 250, 300, 350, 400, 450, 500, 557]

with open(os.path.join(os.getcwd(), "datasets", "test_text.txt"), "r", encoding="utf-8") as f:
    test_text = f.read()

results = calculate_metrics(test_text, vocab, merges, regex_pattern, vocab_sizes)
results_path = os.path.join(folder_path, "results.csv")
results.to_csv(results_path)
plot_path = os.path.join(folder_path, "results.jpg")

print("="*10)
print("Experimental results: \n")
print(results)
print("="*10)
plt = plot_metrics(results, plot_path)
print("Results plots saved")
print("-"*50)
