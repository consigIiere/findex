from pathlib import Path

from findex.index import InvertedIndex
from findex.models import SearchResult
from findex.query import And, Node, Or, Phrase, QueryParser, Term
from findex.reader import iter_documents
from findex.scoring import BaseScorer, BM25Scorer
from findex.utils import extract_snippet, timed


def evaluate_ast(node: Node, index: InvertedIndex) -> set[str]:
    """Рекурсивно обчислює множину doc_id кандидатів за AST деревом запиту."""
    all_doc_ids: set[str] = set(index.doc_lengths.keys())

    if isinstance(node, Term):
        if not node.value:
            return set()
        return {p.doc_id for p in index.get_postings(node.value)}

    if isinstance(node, Phrase):
        if not node.terms:
            return set()
        first_postings = index.get_postings(node.terms[0])
        candidate_docs: set[str] = {p.doc_id for p in first_postings}

        for term in node.terms[1:]:
            postings = index.get_postings(term)
            candidate_docs &= {p.doc_id for p in postings}

        matched_docs: set[str] = set()
        for doc_id in candidate_docs:
            term_positions: list[set[int]] = []
            for term in node.terms:
                postings = index.get_postings(term)
                doc_p = next((p for p in postings if p.doc_id == doc_id), None)
                if doc_p is not None:
                    term_positions.append(set(doc_p.positions))
                else:
                    term_positions.append(set())

            if not term_positions or not term_positions[0]:
                continue

            for pos in term_positions[0]:
                if all((pos + i) in term_positions[i] for i in range(1, len(node.terms))):
                    matched_docs.add(doc_id)
                    break
        return matched_docs

    if isinstance(node, And):
        left_docs = evaluate_ast(node.left, index)
        right_docs = evaluate_ast(node.right, index)
        return left_docs & right_docs

    if isinstance(node, Or):
        left_docs = evaluate_ast(node.left, index)
        right_docs = evaluate_ast(node.right, index)
        return left_docs | right_docs

    child_docs = evaluate_ast(node.child, index)
    return all_doc_ids - child_docs


def extract_terms_from_ast(node: Node) -> list[str]:
    """Витягує список позитивних пошукових слів із дерева AST."""
    terms: list[str] = []

    def _collect(n: Node) -> None:
        if isinstance(n, Term):
            if n.value:
                terms.append(n.value)
        elif isinstance(n, Phrase):
            terms.extend(n.terms)
        elif isinstance(n, (And, Or)):
            _collect(n.left)
            _collect(n.right)
        else:
            pass

    _collect(node)
    return terms


@timed
def ranked_search(
    index: InvertedIndex,
    query_str: str,
    scorer: BaseScorer | None = None,
    top_k: int = 10,
    corpus_dir: str | Path | None = None,
) -> list[SearchResult]:
    """Виконує парсинг запиту, булеву фільтрацію та ранжування результатів."""
    active_scorer: BaseScorer = BM25Scorer() if scorer is None else scorer

    parser = QueryParser(query_str)
    ast_tree: Node = parser.parse()
    candidate_doc_ids: set[str] = evaluate_ast(ast_tree, index)

    if not candidate_doc_ids:
        return []

    terms: list[str] = extract_terms_from_ast(ast_tree)
    if not terms:
        terms = [query_str.lower()]

    results: list[SearchResult] = []
    for doc_id in candidate_doc_ids:
        score: float = active_scorer.score_document(index, doc_id, terms)
        results.append(SearchResult(doc_id=doc_id, score=score))

    results.sort(key=lambda item: item.score, reverse=True)
    top_results: list[SearchResult] = results[:top_k]

    if corpus_dir is not None:
        corpus_path: Path = Path(corpus_dir)
        if corpus_path.is_file() and corpus_path.suffix.lower() == ".jsonl":
            needed_ids = {r.doc_id for r in top_results}
            found_texts: dict[str, str] = {}
            for doc_id, text in iter_documents(corpus_path):
                if doc_id in needed_ids:
                    found_texts[doc_id] = text
                    if len(found_texts) == len(needed_ids):
                        break
            for res in top_results:
                if res.doc_id in found_texts:
                    res.snippet = extract_snippet(found_texts[res.doc_id], terms)
        elif corpus_path.is_dir():
            for res in top_results:
                file_path: Path = corpus_path / res.doc_id
                if file_path.exists():
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                    res.snippet = extract_snippet(text, terms)

    return top_results
