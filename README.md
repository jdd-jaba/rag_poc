# Python Docs RAG

公式の [Python 3 ドキュメント](https://docs.python.org/3/) をベースにしたローカル RAG（Retrieval-Augmented Generation）システムです。すべての処理は自分のマシン上で完結します。クラウド API は不要で、データが外部に送信されることはありません。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│  フェーズ 1 — スクレイピング                                     │
│                                                                 │
│  docs.python.org  ──►  BeautifulSoup  ──►  raw_pages/*.json    │
│      （約1,000ページ）   （テキスト抽出）   （url, title, text） │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  フェーズ 2 — インジェスト                                       │
│                                                                 │
│  raw_pages/*.json  ──►  分割  ──►  埋め込み  ──►  ChromaDB      │
│                       （500字）  （MiniLM-L6）  （ローカル保存） │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  フェーズ 3 — クエリ                                             │
│                                                                 │
│  質問 ──► LLMが言い換え ──► ChromaDB（バリアントごと）           │
│           （3バリアント）    （上位5件ずつ、重複除去）            │
│                                    │                           │
│                                    ▼                           │
│                         llama3.2（ストリーミング）──► 回答      │
└─────────────────────────────────────────────────────────────────┘
```

### 使用コンポーネント

| コンポーネント | ツール | 詳細 |
|---|---|---|
| Webスクレイパー | `aiohttp` + `BeautifulSoup4` | 非同期・10並列・中断再開対応 |
| テキスト分割 | `langchain-text-splitters` | `RecursiveCharacterTextSplitter`、500字・50字オーバーラップ |
| 埋め込みモデル | `sentence-transformers` | `all-MiniLM-L6-v2`、完全オフライン動作（約90MB） |
| ベクトルストア | `ChromaDB` | `./chroma_db/` にローカル永続保存 |
| LLM | `Ollama` + `llama3.2` | ローカル実行、APIキー不要 |
| RAGチェーン | `LangChain` LCEL | マルチクエリ検索 + ストリーミング回答 |

## 事前準備

1. **Python 3.11 以上**

2. **Ollama** — [ollama.com](https://ollama.com) からインストール後、モデルを取得：
   ```bash
   ollama pull llama3.2
   ```

3. **Python依存ライブラリのインストール：**
   ```bash
   pip install -r requirements.txt
   ```

## 実行手順

### ステップ 1 — ドキュメントのスクレイピング

`docs.python.org/3/` 配下の全ページをクロールし、JSONファイルとして保存します。
中断しても安全です。再実行時は取得済みのページは自動でスキップされます。

```bash
python scraper.py
```

出力：`raw_pages/*.json`（約1,000ファイル）と `scraped.log`
所要時間：約2〜3分

### ステップ 2 — ChromaDB へのインジェスト

スクレイピング済みの全ページをチャンクに分割し、埋め込みを生成してベクトルデータベースに保存します。
再実行しても安全です。コレクションが既に存在する場合はスキップされます。

```bash
python ingest.py
```

出力：`chroma_db/` ディレクトリ
所要時間：Apple Silicon で約1〜2分

### ステップ 3 — 質問する

```bash
python query.py "asyncio はどのように動作しますか？"
python query.py "リストとタプルの違いは何ですか？"
python query.py "コンテキストマネージャはどう使いますか？"
```

出力例：
```
Question: asyncio はどのように動作しますか？
------------------------------------------------------------

Generating query variants...

[Query variants (4 total)]
  original: asyncio はどのように動作しますか？
  variant 1: Python の asyncio イベントループとは何ですか？
  variant 2: Python でコルーチンはどのようにスケジュールされますか？
  variant 3: async/await は内部でどのように動作しますか？

[Retrieved chunks for original query]
  score 0.1823 | asyncio — Asynchronous I/O — Python 3.14.3 doc
  score 0.2104 | asyncio-task — Coroutines and Tasks — Python 3...
  score 0.2341 | asyncio-eventloop — Event Loop — Python 3.14.3
  score 0.2789 | library/concurrent.futures — Python 3.14.3 doc
  score 0.3102 | whatsnew/3.11 — What's New In Python 3.11

[Total unique chunks passed to LLM: 14]

------------------------------------------------------------
Answer:

asyncio は async/await 構文を使って並行処理を記述するためのライブラリです。
イベントループを使用してコルーチンを管理・スケジュールします...

Sources:
  - https://docs.python.org/3/library/asyncio.html
  - https://docs.python.org/3/library/asyncio-task.html
```

> **スコアについて：** 値が低いほど関連性が高い（コサイン距離）。`0.2` 未満は強い一致、`0.4` 以上は一致度が低いことを示します。ドキュメントの範囲外の質問をした際の目安になります。

## マルチクエリ検索の仕組み

`"asyncio はどのように動作しますか？"` という質問は、同じ表現を使ったチャンクにしかマッチしません。しかし関連するドキュメントは「イベントループのスケジューリング」「コルーチンの実行モデル」など、異なる表現で書かれている場合があります。

`query.py` では、検索前に LLM が質問を3通りに言い換えます。各バリアントで上位5件のチャンクを取得し、重複を除いたうえで全チャンクを LLM に渡すことで、より広く正確な回答を生成します。

## プロジェクト構成

```
rag_poc/
├── scraper.py          # フェーズ1：非同期Webクローラー
├── ingest.py           # フェーズ2：分割・埋め込み・保存
├── query.py            # フェーズ3：マルチクエリ検索 + ストリーミング回答
├── requirements.txt    # Python依存ライブラリ
├── .gitignore
│
│   # 生成ファイル — gitには含まれません
├── raw_pages/          # スクレイピングしたJSONファイル（1ページ1ファイル）
├── scraped.log         # 取得済みURLの記録（再開用）
└── chroma_db/          # ChromaDB ベクトルストア
```

> `raw_pages/`、`scraped.log`、`chroma_db/` は `.gitignore` で除外されています。
> 上記の3ステップを実行すれば、いつでも再構築できます。
