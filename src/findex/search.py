from functools import lru_cache
import heapq
import os
import re

from findex.index import InvertedIndex, open_index
from findex.models import SearchResult
from findex.query import parse_query
from findex.scoring import BM25Scorer, Scorer, TfidfScorer
from findex.tokenizer import tokenize
from findex.utils import extract_snippet, timed


@timed
def ranked_search(
    index: InvertedIndex,
    query: str,
    scorer: Scorer = BM25Scorer(),
    top_k: int = 5,
    corpus_dir: str | None = None,
) -> list[SearchResult]:
    """
    Виконує ранжований пошук:
    1. Якщо запит містить булеві оператори або фрази в лапках — знаходить відповідні документи через AST-парсер.
    2. Ранжує знайдені документи за обраним скорером (BM25 або TF-IDF).
    3. Відбирає top-k результатів за допомогою heapq.nlargest.
    """
    has_boolean = bool(re.search(r'\b(AND|OR|NOT)\b|[()"]', query))

    if has_boolean:
        ast_tree = parse_query(query)
        candidate_doc_ids = ast_tree.evaluate(index)
    else:
        candidate_doc_ids = None

    terms = list(tokenize(query))
    if not terms:
        return []

    scores: dict[str, float] = {}

    for term in terms:
        postings = index.get_postings(term)
        for p in postings:
            if candidate_doc_ids is not None and p.doc_id not in candidate_doc_ids:
                continue
            term_score = scorer.score_term(term, p.freq, p.doc_id, index)
            scores[p.doc_id] = scores.get(p.doc_id, 0.0) + term_score

    if candidate_doc_ids is not None:
        for doc_id in candidate_doc_ids:
            scores.setdefault(doc_id, 0.0)

    results = [SearchResult(score=score, doc_id=doc_id) for doc_id, score in scores.items()]

    top_results = heapq.nlargest(top_k, results)

    if corpus_dir and os.path.exists(corpus_dir):
        for res in top_results:
            file_path = os.path.join(corpus_dir, res.doc_id)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    res.snippet = extract_snippet(f.read(), terms)

    return top_results


@lru_cache(maxsize=128)
def cached_search_ids(query: str, index_path: str) -> tuple[str, ...]:
    """Кешує ідентифікатори знайдених файлів для уникнення повторних обрахунків."""
    with open_index(index_path) as idx:
        results = ranked_search(idx, query, top_k=5)
        return tuple(r.doc_id for r in results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ранжований та булев пошук у документах (BM25 / TF-IDF)")
    parser.add_argument("index_path", help="Шлях до бінарного файлу індексу (.bin)")
    parser.add_argument("query", help="Пошуковий запит (підтримує AND, OR, NOT, дужки та фрази в лапках)")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Модель ранжування (за замовчуванням: bm25)")
    parser.add_argument("-k", type=int, default=5, help="Кількість результатів (за замовчуванням: 5)")
    parser.add_argument("--corpus", default="data", help="Шлях до папки з текстами для створення сніпетів")

    args = parser.parse_args()

    active_scorer = BM25Scorer() if args.scorer == "bm25" else TfidfScorer()

    with open_index(args.index_path) as index:
        results = ranked_search(
            index=index,
            query=args.query,
            scorer=active_scorer,
            top_k=args.k,
            corpus_dir=args.corpus,
        )

        print(f"\nЗнайдено результатів (топ-{args.k}) за моделлю {args.scorer.upper()}:")
        print("-" * 60)
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res.score:.4f}] {res.doc_id}")
            if res.snippet:
                print(f"   Сніпет: {res.snippet}")
            print("-" * 60)