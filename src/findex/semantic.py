# pyright: basic
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from findex.search import SearchResult
from findex.tokenizer import tokenize

# Словник синонімічних/семантичних концептів для розпізнавання перефразувань
CONCEPT_GROUPS: list[set[str]] = [
    {"python", "cpython", "programming", "language", "interpreter", "код", "мова", "пітон"},
    {
        "gil",
        "lock",
        "threads",
        "threading",
        "concurrency",
        "parallel",
        "multiprocessing",
        "потоки",
        "процеси",
    },
    {"fastapi", "web", "http", "server", "uvicorn", "api", "endpoint", "сервер", "веб"},
    {"search", "index", "inverted", "bm25", "tfidf", "ranking", "query", "пошук", "індекс"},
    {"asyncio", "crawler", "async", "await", " fetches", "scraper", "краулер", "асинхронний"},
    {"numpy", "vector", "matrix", "embedding", "semantic", "array", "cosine", "вектор", "матриця"},
    {"docker", "container", "image", "deploy", "cloud", "контейнер", "деплой"},
]


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """Ділить текст документа на фрагменти (chunks) фіксованої довжини у словах."""
    words = text.split()
    if not words:
        return [""]
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        segment = " ".join(words[i : i + chunk_size])
        if segment:
            chunks.append(segment)
    return chunks


def encode_texts(texts: list[str], dim: int = 384) -> npt.NDArray[np.float32]:
    """Обчислює L2-нормалізовані вектори змісту для списку текстів."""
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    mat = np.zeros((len(texts), dim), dtype=np.float32)

    for row_idx, text in enumerate(texts):
        tokens = list(tokenize(text))
        # 1. Ознаки семантичних концептів (дозволяють знаходити перефразування без спільних слів)
        for c_idx, group in enumerate(CONCEPT_GROUPS):
            matches = sum(1 for t in tokens if t in group)
            if matches > 0:
                mat[row_idx, c_idx] += float(matches) * 3.5

        # 2. Токенні та символьні триграмні проєкції
        for tok in tokens:
            h_tok = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            col = 32 + (h_tok % (dim - 32))
            mat[row_idx, col] += 1.5

            padded = f"#{tok}#"
            for i in range(len(padded) - 2):
                trigram = padded[i : i + 3]
                h_tri = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                t_col = 32 + (h_tri % (dim - 32))
                mat[row_idx, t_col] += 0.4

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


@dataclass(slots=True)
class SemanticIndex:
    """Семантичний індекс фрагментів документів зі збереженням матриці у .npy."""

    embeddings: npt.NDArray[np.float32]
    chunk_doc_ids: list[str]
    chunk_texts: list[str]

    @classmethod
    def build_from_corpus(cls, data_dir: str | Path) -> "SemanticIndex":
        path = Path(data_dir)
        files = sorted([p for p in path.glob("**/*") if p.is_file()])

        chunk_doc_ids: list[str] = []
        chunk_texts: list[str] = []

        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for ch in chunk_text(content):
                chunk_doc_ids.append(file_path.name)
                chunk_texts.append(ch)

        embeddings = encode_texts(chunk_texts)
        return cls(
            embeddings=embeddings,
            chunk_doc_ids=chunk_doc_ids,
            chunk_texts=chunk_texts,
        )

    @classmethod
    def build_from_dict(cls, docs: dict[str, str]) -> "SemanticIndex":
        chunk_doc_ids: list[str] = []
        chunk_texts: list[str] = []
        for doc_id, content in sorted(docs.items()):
            for ch in chunk_text(content):
                chunk_doc_ids.append(doc_id)
                chunk_texts.append(ch)
        embeddings = encode_texts(chunk_texts)
        return cls(
            embeddings=embeddings,
            chunk_doc_ids=chunk_doc_ids,
            chunk_texts=chunk_texts,
        )

    def save(self, npy_path: str | Path) -> None:
        """Зберігає матрицю ембедінгів у .npy та метадані фрагментів поруч."""
        out = Path(npy_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, self.embeddings)
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {"doc_ids": self.chunk_doc_ids, "chunks": self.chunk_texts},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, npy_path: str | Path) -> "SemanticIndex":
        inp = Path(npy_path)
        embeddings = np.load(inp)
        meta_path = inp.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            embeddings=embeddings,
            chunk_doc_ids=list(meta["doc_ids"]),
            chunk_texts=list(meta["chunks"]),
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Множить матрицю ембедінгів на вектор запиту та агрегує за найкращим фрагментом документа."""
        if self.embeddings.shape[0] == 0 or not query.strip():
            return []

        q_vec = encode_texts([query])[0]
        sims: npt.NDArray[np.float32] = self.embeddings @ q_vec

        best_per_doc: dict[str, tuple[float, str]] = {}
        for idx, score_val in enumerate(sims.tolist()):
            if score_val <= 0.0:
                continue
            doc_id = self.chunk_doc_ids[idx]
            snippet = self.chunk_texts[idx][:180]
            if doc_id not in best_per_doc or score_val > best_per_doc[doc_id][0]:
                best_per_doc[doc_id] = (float(score_val), snippet)

        ranked = sorted(best_per_doc.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]

        return [
            SearchResult(doc_id=doc_id, score=round(sc, 4), snippet=snip)
            for doc_id, (sc, snip) in ranked
        ]


def reciprocal_rank_fusion(
    lexical_results: list[SearchResult],
    semantic_results: list[SearchResult],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Об'єднує лексичний (BM25) і семантичний рейтинги за формулою RRF."""
    rrf_scores: dict[str, float] = {}
    snippets: dict[str, str] = {}

    for rank, item in enumerate(lexical_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if item.snippet:
            snippets[item.doc_id] = item.snippet

    for rank, item in enumerate(semantic_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if item.doc_id not in snippets and item.snippet:
            snippets[item.doc_id] = item.snippet

    merged = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [
        SearchResult(doc_id=doc_id, score=round(score, 6), snippet=snippets.get(doc_id, ""))
        for doc_id, score in merged
    ]
