import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.reader import iter_documents
from findex.tokenizer import tokenize


def run_streaming(corpus_path: str | Path):

    tracemalloc.start()
    start_time = time.perf_counter()

    doc_count = 0
    total_tokens = 0
    vocab = Counter()

    for _, text in iter_documents(corpus_path):
        doc_count += 1
        for token in tokenize(text):
            total_tokens += 1
            vocab[token] += 1

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": "Streaming (Generators)",
        "docs": doc_count,
        "tokens": total_tokens,
        "unique_vocab": len(vocab),
        "top_50": vocab.most_common(50),
        "time_sec": elapsed_time,
        "peak_mem_mb": peak_memory / (1024 * 1024),
    }


def run_eager(corpus_path: str | Path):

    tracemalloc.start()
    start_time = time.perf_counter()

    path = Path(corpus_path)
    all_files = list(path.rglob("*.txt"))


    raw_texts = [f.read_text(encoding="utf-8", errors="replace") for f in all_files]


    all_tokens = []
    for text in raw_texts:
        all_tokens.extend(list(tokenize(text)))

    doc_count = len(raw_texts)
    total_tokens = len(all_tokens)
    vocab = Counter(all_tokens)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": "Eager (Lists)",
        "docs": doc_count,
        "tokens": total_tokens,
        "unique_vocab": len(vocab),
        "top_50": vocab.most_common(50),
        "time_sec": elapsed_time,
        "peak_mem_mb": peak_memory / (1024 * 1024),
    }


if __name__ == "__main__":
    corpus = "data"

    print("=== Запуск Streaming версії ===")
    s_res = run_streaming(corpus)
    print(f"Час: {s_res['time_sec']:.3f} с | Пікова пам'ять: {s_res['peak_mem_mb']:.2f} МБ")
    print(f"Документів: {s_res['docs']} | Токенів: {s_res['tokens']} | Словник: {s_res['unique_vocab']}")

    print("\n=== Запуск Eager версії ===")
    e_res = run_eager(corpus)
    print(f"Час: {e_res['time_sec']:.3f} с | Пікова пам'ять: {e_res['peak_mem_mb']:.2f} МБ")
    print(f"Документів: {e_res['docs']} | Токенів: {e_res['tokens']} | Словник: {e_res['unique_vocab']}")

    print("\n=== Топ-10 найчастіших слів корпусу ===")
    for word, count in s_res["top_50"][:10]:
        print(f"{word}: {count}")