# Tokenizer
Code written while learning Tokenization, all code written to make DataCollector utility, and a final project.

### Folder Overview:
- `app/`- Code for the website.
- `datasets/`- Folders of datasets made using the "DataCollector" class, includes an `output.txt` and an `input.txt`.
- `notebooks/` - Jupyter notebooks for experiments and analysis.
- `output_files/` - Generated tokenizer/model outputs.
- `scripts/` - Data collection, training, and utility scripts.

### Important Files
- `scripts/train.py`- Naive Tokenizer.
- `scripts/train_inverted.py` - Optimized Tokenizer.
- `scripts/experiments.py` -Benchmarks each tokenizer with variable vocabulary size.
- `scripts/data_collection.py`- Contains the "DataCollector" class to collect data of any size with a simple script.
- `scripts/make_data.py` - Calls the data class, and uses all six natural languages, and five programming languages to make a dataset. Automatically dives any **x** MB among different sources.
- `scripts/single_run.py` - Uses the data, regex, vocab_size, trains the tokenizer using **InvertedTokenizer**, benchmarks it.

### Live Link
- [https://tokenizer-kcae.onrender.com/]

### Notes
- [Planning to include a token visualizer that shows how a final token was formed through the merges, and what subtokens it inherited.]