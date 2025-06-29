from flask import Flask, render_template, request, jsonify
import json
import os
import sys
import numpy as np
from pathlib import Path
import datetime
import difflib
from functools import lru_cache

# Add bm25s to path
sys.path.append(str(Path(__file__).parent.parent / 'bm25s'))

from bm25s import BM25
from bm25s.tokenization import tokenize

app = Flask(__name__)

# Global variables untuk menyimpan data
bm25_model = None
corpus = []
documents = []
indonesian_stopwords = []

# Inisialisasi stemmer - prioritas Sastrawi untuk bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    sastrawi_stemmer = factory.create_stemmer()
    
    # Wrapper untuk Sastrawi agar kompatibel dengan bm25s
    class SastrawiWrapper:
        def __init__(self, stemmer):
            self.stemmer = stemmer
        
        def __call__(self, words):
            if isinstance(words, str):
                return self.stemmer.stem(words)
            elif isinstance(words, list):
                return [self.stemmer.stem(word) for word in words]
            else:
                return words
    
    stemmer = SastrawiWrapper(sastrawi_stemmer)
    print("Using Sastrawi for Indonesian stemming")
    USE_PYSTEMMER = False
except ImportError:
    try:
        import Stemmer
        # PyStemmer tidak mendukung bahasa Indonesia, jadi gunakan English stemmer
        # Ini masih lebih cepat dari tanpa stemming untuk preprocessing
        stemmer = Stemmer.Stemmer("english")
        print("Using PyStemmer (English) as fallback - faster but not Indonesian")
        USE_PYSTEMMER = True
    except ImportError:
        stemmer = None
        print("No stemmer available - proceeding without stemming")
        USE_PYSTEMMER = False

@lru_cache(maxsize=100_000)
def cached_stem(word):
    if stemmer is None:
        return word
    if USE_PYSTEMMER:
        # PyStemmer menggunakan stemWord method
        return stemmer.stemWord(word)
    else:
        # Sastrawi menggunakan wrapper
        return stemmer(word)

def stem_list(words):
    if stemmer is None:
        return words
    if USE_PYSTEMMER:
        # PyStemmer lebih efisien untuk batch processing
        return stemmer.stemWords(words)
    else:
        # Sastrawi menggunakan wrapper
        return stemmer(words)

def load_indonesian_stopwords():
    """Load stopwords Indonesia dari file"""
    global indonesian_stopwords
    stopwords_path = Path(__file__).parent.parent / 'bm25s' / 'stopwords-id.txt'
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            indonesian_stopwords = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(indonesian_stopwords)} Indonesian stopwords")
    except FileNotFoundError:
        print("Warning: stopwords-id.txt not found, using empty stopwords list")
        indonesian_stopwords = []

def load_data():
    """Load data dari file JSONL (semua dokumen)"""
    global corpus, documents
    data_dir = Path(__file__).parent / 'data'
    corpus = []
    documents = []
    seen_ids = set()  # Untuk tracking duplikat
    
    for file_path in data_dir.glob('train.01.jsonl'):
        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    doc_id = doc.get('id', '')
                    
                    # Skip jika ID sudah ada (deduplication)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    
                    corpus.append(doc)
                    
                    # Cek kelengkapan data
                    if not (
                        doc.get('id') and
                        doc.get('category') and
                        doc.get('paragraphs') and len(doc.get('paragraphs')) > 0 and
                        doc.get('source') and
                        doc.get('source_url') and
                        doc.get('summary') and len(doc.get('summary')) > 0
                    ):
                        continue  # Lewati dokumen yang tidak lengkap

                    # Gabungkan semua text
                    text_parts = []
                    for paragraph in doc.get('paragraphs', []):
                        for sentence in paragraph:
                            text_parts.append(' '.join(sentence))
                    full_text = ' '.join(text_parts)

                    # Gunakan ID sebagai judul, tapi format agar lebih readable
                    raw_id = doc.get('id', '')
                    # Hilangkan angka di depan dan ganti '-' dengan spasi
                    title = raw_id
                    date_str = ''
                    date_obj = None
                    if '-' in raw_id:
                        parts = raw_id.split('-')
                        # Ambil timestamp
                        try:
                            ts = int(parts[0])
                            dt = datetime.datetime.fromtimestamp(ts)
                            date_obj = ts
                            # Format tanggal Indonesia
                            bulan = [
                                '', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                                'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
                            ]
                            date_str = f"{dt.day} {bulan[dt.month]} {dt.year}"
                        except Exception:
                            date_str = ''
                            date_obj = None
                        title = '-'.join(parts[1:])  # hapus bagian angka timestamp di depan
                    title = title.replace('-', ' ').strip()

                    documents.append({
                        'id': doc.get('id', ''),
                        'title': title,
                        'category': doc.get('category', ''),
                        'text': full_text,
                        'source': doc.get('source', ''),
                        'source_url': doc.get('source_url', ''),
                        'summary': ' '.join([' '.join(sentence) for sentence in doc.get("summary", [])]),
                        'date': date_str,
                        'date_obj': date_obj
                    })
    
    print(f"Loaded {len(documents)} unique documents (removed duplicates)")
    print(f"Total unique IDs found: {len(seen_ids)}")

INDEX_DIR = Path(__file__).parent.parent / 'bm25_index'

def build_bm25_index():
    global bm25_model
    if os.path.exists(INDEX_DIR):
        print('Loading BM25 index from file...')
        bm25_model = BM25.load(INDEX_DIR)
    else:
        print('Building BM25 index...')
        
        # Gunakan stemmer yang sudah diinisialisasi (prioritas Sastrawi)
        if stemmer is not None:
            if USE_PYSTEMMER:
                print("Using PyStemmer (English) for fast stemming...")
            else:
                print("Using Sastrawi for Indonesian stemming...")
            stemmer_bm25s = stemmer
        else:
            print("No stemming available...")
            stemmer_bm25s = None
        
        # Tokenize dengan built-in bm25s stemming
        tokenized_docs = tokenize(
            [doc['text'] for doc in documents], 
            stopwords=indonesian_stopwords, 
            stemmer=stemmer_bm25s, 
            return_ids=False, 
            show_progress=True
        )
        
        bm25_model = BM25(k1=1.5, b=0.75, method='lucene')
        bm25_model.index(tokenized_docs, show_progress=True)
        bm25_model.save(INDEX_DIR)
        print('BM25 index built and saved.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'results': [], 'total': 0})
    
    sort = request.args.get('sort', 'relevansi')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Gunakan stemming yang konsisten dengan indexing
    stemmer_bm25s = stemmer if stemmer else None
    
    # Tokenize query dengan stemming yang sama
    query_tokens = tokenize([query], stopwords=indonesian_stopwords, stemmer=stemmer_bm25s, return_ids=False, show_progress=False)[0]
    print("Query tokens setelah stemming:", query_tokens)
    
    # Tentukan jumlah dokumen di index
    num_docs = bm25_model.scores['num_docs'] if hasattr(bm25_model, 'scores') and 'num_docs' in bm25_model.scores else len(documents)
    k = min(20, num_docs)
    
    # Search menggunakan BM25
    results = bm25_model.retrieve([query_tokens], k=k, sorted=True)
    
    # Format results
    search_results = []
    for doc_idx, score in zip(results.documents[0], results.scores[0]):
        doc = documents[doc_idx]
        search_results.append({
            'id': doc['id'],
            'title': doc['title'],
            'category': doc['category'],
            'text': doc['text'][:300] + '...' if len(doc['text']) > 300 else doc['text'],
            'summary': doc['summary'],
            'source': doc['source'],
            'source_url': doc['source_url'],
            'score': float(score),
            'date': doc.get('date', ''),
            'date_obj': doc.get('date_obj', None)
        })
    
    # Setelah search_results selesai dibuat, lakukan sorting jika perlu
    if sort == 'terbaru':
        search_results.sort(key=lambda x: x['date_obj'] if x['date_obj'] is not None else 0, reverse=True)
    elif sort == 'terlama':
        search_results.sort(key=lambda x: x['date_obj'] if x['date_obj'] is not None else 0)
    
    SCORE_THRESHOLD = 0.01
    # Filter hasil berdasarkan skor minimum
    filtered_results = [r for r in search_results if r['score'] > SCORE_THRESHOLD]
    if len(filtered_results) == 0:
        paginated_results = []
        total_results = 0
        total_pages = 1
    else:
        total_results = len(filtered_results)
        total_pages = (total_results + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = filtered_results[start:end]
    
    suggestion = None
    if len(search_results) == 0:
        # Kumpulkan semua kata unik dari judul, kategori, dan sumber
        all_terms = set()
        for doc in documents:
            all_terms.update(doc['title'].split())
            all_terms.add(doc['category'])
            all_terms.add(doc['source'])
        # Cari suggestion terdekat
        matches = difflib.get_close_matches(query, all_terms, n=1, cutoff=0.7)
        if matches:
            suggestion = matches[0]
    
    # Ambil semua kategori unik
    all_categories = sorted({doc['category'] for doc in documents})
    category_filter = request.args.get('category', '').strip()
    # Filter hasil berdasarkan kategori jika ada
    if category_filter:
        paginated_results = [r for r in paginated_results if r['category'].lower() == category_filter.lower()]
    
    return jsonify({
        'results': paginated_results,
        'total': total_results,
        'query': query,
        'page': page,
        'total_pages': total_pages,
        'suggestion': suggestion,
        'categories': all_categories,
        'selected_category': category_filter,
        'used_keywords': query_tokens  # token query tanpa stopwords
    })

@app.route('/api/search')
def api_search():
    """API endpoint untuk search"""
    return search()

if __name__ == '__main__':
    print("Loading Indonesian stopwords...")
    load_indonesian_stopwords()
    
    print("Loading data...")
    load_data()
    
    print("Building BM25 index...")
    build_bm25_index()
    
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000) 