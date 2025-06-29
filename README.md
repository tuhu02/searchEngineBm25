# BM25 Search Engine

Web search engine menggunakan algoritma BM25 untuk pencarian dokumen berita Indonesia.

## Fitur

- 🔍 Pencarian menggunakan algoritma BM25
- 📰 Dataset berita Indonesia dari berbagai sumber
- 🌐 Interface web yang modern dan responsif
- ⚡ Performa pencarian yang cepat
- 📊 Menampilkan score relevansi untuk setiap hasil

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pastikan folder `bm25s` berada di direktori parent (satu level di atas folder `uas`)

## Penggunaan

1. Jalankan aplikasi:
```bash
python app.py
```

2. Buka browser dan akses `http://localhost:5000`

3. Masukkan kata kunci pencarian dan tekan Enter atau klik tombol "Cari"

## Struktur Data

Aplikasi menggunakan data dari file JSONL yang berisi:
- `id`: ID unik dokumen
- `category`: Kategori berita
- `paragraphs`: Paragraf-paragraf berita
- `source`: Sumber berita
- `source_url`: URL sumber berita
- `summary`: Ringkasan berita

## API Endpoints

- `GET /`: Halaman utama search engine
- `GET /search?q=<query>`: Endpoint pencarian
- `GET /api/search?q=<query>`: API endpoint untuk pencarian

## Algoritma BM25

Aplikasi menggunakan implementasi BM25 dengan parameter:
- `k1 = 1.5`: Parameter untuk term frequency
- `b = 0.75`: Parameter untuk document length normalization
- `method = 'lucene'`: Metode scoring yang digunakan

## Teknologi

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Search Algorithm**: BM25
- **Data Processing**: NumPy, SciPy 