'''
Provides a class to collect text data for Languages (Normal and Programming).

desired output:

obj1 = DataCollector()
objj1.add_language("english", size=100) # collects 100MB of English language data.
obj1.add_programming_language("python", size=100) # collects 100MB of Python code.

Provides the text in either a single text file, or a folder with a seperate file for each language and programming language.
'''

import random
import requests
from datasets import load_dataset
from tqdm import tqdm

class DataCollector:
    
    def __init__(self, output_file_path, report_file_path):
        
        self.output_path = output_file_path
        self.report_path = report_file_path
        
        self.wikipedia_language_keymap = {
            
            "english": "en",
            "hindi": "hi",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "japanese": "ja"
        }
        
        self.github_repos = {
            
            "python": [
                "python/cpython",
                "numpy/numpy",
                "pallets/flask",
                "psf/requests",
                "django/django",
            ],

            "cpp": [
                "tensorflow/tensorflow",
                "opencv/opencv",
                "llvm/llvm-project",
                "bitcoin/bitcoin",
                "catchorg/Catch2",
            ],

            "javascript": [
                "facebook/react",
                "nodejs/node",
                "expressjs/express",
                "axios/axios",
                "vuejs/core",
            ],

            "java": [
                "spring-projects/spring-framework",
                "apache/kafka",
                "elastic/elasticsearch",
                "google/guava",
                "apache/maven",
            ],

            "rust": [
                "rust-lang/rust",
                "tokio-rs/tokio",
                "serde-rs/serde",
                "clap-rs/clap",
                "BurntSushi/ripgrep",
            ],
        }
        
        self.programming_extensions = {
            
            "python": {"py"},
            
            "cpp": {"cpp", "cc", "cxx", "h", "hpp"},
            
            "javascript": {"js", "jsx", "mjs", "cjs"},
            
            "java": {"java"},
            
            "rust": {"rs"},
            
        }

        
        self.news_feeds = {
            
            "english": [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://feeds.skynews.com/feeds/rss/home.xml"
            ],

            "hindi": [
                "https://feeds.bbci.co.uk/hindi/rss.xml",
                "https://www.republicbharat.com/rss/india.xml"
            ],

            "japanese": [
                "https://feeds.bbci.co.uk/japanese/rss.xml",
                "https://rss.asahi.com/rss/asahi/newsheadlines.rdf"
            ],

            "french": [
                "https://www.france24.com/fr/rss",
                "https://www.mediapart.fr/articles/feed"
                
            ],

            "german": [
                "https://www.deutschland.de/de/feed-news/rss.xml",
                "https://newsfeed.zeit.de/index"
            ],

            "spanish": [
                "https://feeds.bbci.co.uk/mundo/rss.xml",
                "https://e00-elmundo.uecdn.es/rss/portada.xml"
            ],
        }
        
    def add_language(self, language, size, random_seed):
        
        '''
        Adds `size` MB of language.
        '''
        
        import feedparser
        from newspaper import Article
        import nltk
        
        nltk.download("punkt")
        nltk.download("punkt_tab")
        
        rss_urls = self.news_feeds[language]
        self.url_seed = random.Random(random_seed)
        
        self.url_seed.shuffle(rss_urls)
        
        articles = []
        
        for url in tqdm(rss_urls, desc="Iterating over news RSS feeds:"):
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                article = Article(entry.link)
                
                try:
                    article.download()
                    article.parse()
                    
                    article.nlp()
                    
                    parsed = {
                        "title": article.title,
                        "text": article.text,
                        "keywords": article.keywords
                    }
                    
                    articles.append(parsed)
                    
                except Exception as e:
                    print(f"Error processing article {entry.link}: {str(e)}")
                    
        combined_text = ""
        for article in articles:
            combined_text += article.get("title", "")
            combined_text += article.get("text", "")
            combined_text += " ".join(article.get("keywords", ""))
            
        total_bytes_in_mb = (len(combined_text.encode("utf-8")) / (1024 * 1024))
        
        if total_bytes_in_mb <= (size // 2):
            remaining_size = (size  - total_bytes_in_mb)
            
        elif total_bytes_in_mb > (size // 2):
            # if size > x//2, we only take x//2
            encoded = combined_text.encode("utf-8")
            combined_text = encoded[:((size // 2) * 1024 * 1024)].decode(
                "utf-8",
                errors="ignore"
            )
            
            remaining_size = (size // 2)
            
        with open(self.report_path, "a") as f:
            f.write(f"Added news article in {language} with text size {total_bytes_in_mb:0.2f} MB.\n")
            f.close()
            
        country_code_wikipedia = self.wikipedia_language_keymap[language]
        shard_code = f"20231101.{country_code_wikipedia}"
        
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": "wikimedia/wikipedia"}
        )
        
        data = response.json()
        files = [f for f in data["parquet_files"] if f["config"] == shard_code]
        
        file_indices = list(range(len(files)))
        
        file_shuffle_random_seed = random.Random(random_seed + 1)
        file_shuffle_random_seed.shuffle(file_indices)
        
        target_bytes = (remaining_size * 1024 * 1024)
        all_shards = [f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/{shard_code}/train/{idx:04d}.parquet" for idx in file_indices]
        
        row_seed = random_seed + 2
        collected_bytes = 0
        
        for shard in tqdm(all_shards, desc="Iterating over Wikipedia Shards:"):
            
            if collected_bytes >= target_bytes:
                break
            
            dataset = load_dataset(
                "parquet",
                data_files={
                    "train": shard
                },
                split="train",
                streaming=True
            )
            
            dataset = dataset.shuffle(
                seed=row_seed,
                buffer_size=10_000
            )
            
            for row in dataset:
                text = row["text"]
                title = row["title"]
                
                if not text:
                    continue
                
                text_bytes = len(text.encode("utf-8"))
                title_bytes = len(title.encode("utf-8"))
                
                combined_text += text
                combined_text += title
                
                collected_bytes += text_bytes
                collected_bytes += title_bytes
                
                if collected_bytes >= target_bytes:
                    break
        
        with open(self.report_path, "a") as f:
            f.write(f"Added wikipedia data in {language} with text size {(collected_bytes / (1024 * 1024)):0.2f} MB.\n")
            f.close()        
        
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(combined_text)
            f.write("\n\n")
            f.close()
            
        print(f"Added {language} to data file.")
        
    def add_programming_language(self, language, size, random_seed):
        
        '''
        Adds `size` MB of programming language.
        '''
        
        target_bytes = (size * 1024 * 1024)
        
        headers = {
            "Accept": "application/vnd.github+json",
            "X-Github-Api-Version": "2026-03-10",
        }
        
        repositories = self.github_repos[language].copy()
        repo_seed = random.Random(random_seed)
        file_seed = random.Random(random_seed + 1)
        
        repo_seed.shuffle(repositories)
        extensions = self.programming_extensions[language]
        
        collected_files = []
        collected_bytes = 0
        
        for repository in tqdm(repositories, desc="Iterating over Repositories:"):
            
            if collected_bytes >= target_bytes:
                break
            
            owner, repo = repository.split("/")
            
            repo_url = (
                f"https://api.github.com/repos/"
                f"{owner}/{repo}"
            )
            
            response = requests.get(
                repo_url,
                headers=headers
            )
            response.raise_for_status()
            
            repo_data = response.json()
            default_branch = repo_data["default_branch"]
        
            tree_url = (
                f"https://api.github.com/repos/"
                f"{owner}/{repo}/git/trees/"
                f"{default_branch}"
            )
            
            response = requests.get(tree_url, params={"recursive": "1"}, headers=headers)
            response.raise_for_status()
            
            tree_data = response.json()
            
            if tree_data.get("truncated", False):
                print("Warning: repository tree was truncated.")
                
            files = []
            
            for item in tree_data.get("tree", []):
                
                if item["type"] != "blob":
                    continue
                
                path = item["path"]
                
                if "." in path:
                    extension = path.rsplit(".", 1)[1]
                    
                if extension.lower() not in extensions:
                    continue
                
                files.append(path)
                file_seed.shuffle(files)
                
                for path in files:
                    
                    if collected_bytes >= target_bytes:
                        break
                    
                    raw_url = (
                        f"https://raw.githubusercontent.com/"
                        f"{owner}/{repo}/"
                        f"{default_branch}/{path}"
                    )
                    
                    response = requests.get(raw_url, headers=headers)
                    
                    if response.status_code != 200:
                        print(f"Skipping {path} because of HTTP {response.status_code}")
                        
                    content = response.content
                    
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        print(f"Skipping non-utf-8 file: {path}")
                        continue
                    
                    if not text.strip():
                        continue
                    
                    bytes_ = len(text.encode("utf-8"))
                    remaining = target_bytes - collected_bytes
                    
                    if bytes_ > remaining:
                        
                        encoded = text.encode("utf-8")
                        text = encoded[:remaining].decode("utf-8", errors="ignore")
                        collected_bytes += remaining
                        
                    collected_files.append(text)
                    collected_bytes += bytes_
                    
        corpus = "\n\n".join(collected_files)
        
        with open(self.report_path, "a") as f:
            f.write(f"Added github data for {language} with text size {(collected_bytes / (1024 * 1024)):0.2f} MB.\n")
            f.close()
            
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(corpus)
            f.write("\n\n")
            f.close()    
                        

