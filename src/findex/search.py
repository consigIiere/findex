import argparse
import time

from findex.index import InvertedIndex


def intersect_postings(list1: list[str], list2: list[str]) -> list[str]:
    """
    Алгоритм злиття двох відсортованих списків doc_id (Two-Pointer merge).
    Працює за лінійний час O(len(list1) + len(list2)) без виділення додаткової пам'яті.
    """
    result = []
    i = 0
    j = 0

    while i < len(list1) and j < len(list2):
        doc1 = list1[i]
        doc2 = list2[j]

        if doc1 == doc2:
            result.append(doc1)
            i += 1
            j += 1
        elif doc1 < doc2:
            i += 1
        else:
            j += 1

    return result


def intersect_sets(list1: list[str], list2: list[str]) -> list[str]:
    """Перетин списків через стандартний тип set (виділяє пам'ять під хеш-таблиці)."""
    return sorted(list(set(list1) & set(list2)))


def search_boolean(
    index: InvertedIndex,
    query: str,
    use_merge: bool = True
) -> list[str]:
    """
    Виконує булевий пошук (підтримує окремі слова або комбінації 'term1 AND term2').
    """
    raw_terms = [part.strip() for part in query.split("AND") if part.strip()]
    if not raw_terms:
        return []

    doc_lists = []
    for term in raw_terms:
        postings = index.get_postings(term)
        docs = sorted([p.doc_id for p in postings])
        doc_lists.append(docs)

    if any(len(lst) == 0 for lst in doc_lists):
        return []

    doc_lists.sort(key=len)

    intersect_func = intersect_postings if use_merge else intersect_sets
    current_result = doc_lists[0]

    for next_list in doc_lists[1:]:
        current_result = intersect_func(current_result, next_list)
        if not current_result:
            break

    return current_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Швидкий булевий пошук за індексом")
    parser.add_argument("index_path", type=str, help="Шлях до бінарного файлу індексу (index.bin)")
    parser.add_argument("query", type=str, help="Пошуковий запит (наприклад, 'python AND code')")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Алгоритм перетину")
    args = parser.parse_args()

    start_load = time.perf_counter()
    index = InvertedIndex.load(args.index_path)
    load_time = time.perf_counter() - start_load
    print(f"Індекс завантажено за {load_time:.4f} с.")

    use_merge = (args.engine == "merge")
    start_search = time.perf_counter()
    results = search_boolean(index, args.query, use_merge=use_merge)
    search_time = time.perf_counter() - start_search

    print(f"\nРезультати для запиту: '{args.query}' (двигун: {args.engine})")
    print(f"Знайдено документів: {len(results)} (час: {search_time * 1000:.3f} мс)")
    for doc in results[:10]:
        print(f" - {doc}")
    if len(results) > 10:
        print(f" ... і ще {len(results) - 10} документів")