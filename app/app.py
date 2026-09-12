from flask import Flask, render_template, redirect, request, url_for
from utils import encode_decode
import time

app = Flask(__name__)

available_tokenizers = {"100MB_GPT4"}

@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "POST":
        vocab_size = request.form.get("vocab")
        tokenizer = request.form.get("tokenizer")
        text = request.form.get("text")
        
        vocab_size = int(vocab_size)
        if vocab_size < 256:
            vocab_size = 256
            
        if vocab_size > 50257:
            vocab_size = 50257
        
        if tokenizer not in available_tokenizers:
            return render_template('index.html', error="This Tokenizer has not yet been made available.")
        else:
            
            start_time = time.perf_counter()
            results_dict = encode_decode(text, vocab_size, tokenizer)
            end_time = time.perf_counter()
            
            return render_template('index.html', results=results_dict, total_time=(end_time -  start_time), length=len(text))
        
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)