from abc import ABC, abstractmethod
import re
from typing import Set

from findex.index import InvertedIndex
from findex.tokenizer import tokenize


# --- Вузли дерева абстрактного синтаксису (AST) ---

class Node(ABC):
    """Базовий абстрактний вузол синтаксичного дерева запиту."""

    @abstractmethod
    def evaluate(self, index: InvertedIndex) -> set[str]:
        """Повертає множину ідентифікаторів документів (doc_id), що задовольняють умову."""
        pass


class Term(Node):
    """Вузол одиночного терма."""

    def __init__(self, value: str) -> None:
        self.value = value.casefold()

    def evaluate(self, index: InvertedIndex) -> set[str]:
        postings = index.get_postings(self.value)
        return {p.doc_id for p in postings}

    def __repr__(self) -> str:
        return f"Term({self.value!r})"


class Phrase(Node):
    """
    Вузол точної фрази у лапках: "event loop".
    Перевіряє, щоб слова йшли послідовно одне за одним за збереженими позиціями.
    """

    def __init__(self, terms: list[str]) -> None:
        self.terms = [t.casefold() for t in terms]

    def evaluate(self, index: InvertedIndex) -> set[str]:
        if not self.terms:
            return set()

        first_postings = {p.doc_id: p.positions for p in index.get_postings(self.terms[0])}
        matching_docs: set[str] = set(first_postings.keys())

        for offset, term in enumerate(self.terms[1:], start=1):
            term_postings = {p.doc_id: set(p.positions) for p in index.get_postings(term)}
            matching_docs &= term_postings.keys()

            valid_in_this_step: set[str] = set()
            for doc_id in matching_docs:
                prev_positions = first_postings[doc_id]
                curr_positions = term_postings[doc_id]
                # Перевіряємо наявність позиції pos + offset
                if any((pos + offset) in curr_positions for pos in prev_positions):
                    valid_in_this_step.add(doc_id)
            matching_docs = valid_in_this_step

        return matching_docs

    def __repr__(self) -> str:
        return f"Phrase({self.terms!r})"


class And(Node):
    """Логічне І: перетин результатів лівого і правого підвиразів."""

    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def evaluate(self, index: InvertedIndex) -> set[str]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def __repr__(self) -> str:
        return f"And({self.left!r}, {self.right!r})"


class Or(Node):
    """Логічне АБО: об'єднання результатів лівого і правого підвиразів."""

    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def evaluate(self, index: InvertedIndex) -> set[str]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def __repr__(self) -> str:
        return f"Or({self.left!r}, {self.right!r})"


class Not(Node):
    """Логічне НІ: усі документи корпусу мінус документи підвузла."""

    def __init__(self, child: Node) -> None:
        self.child = child

    def evaluate(self, index: InvertedIndex) -> set[str]:
        all_docs = set(index.docs.keys())
        return all_docs - self.child.evaluate(index)

    def __repr__(self) -> str:
        return f"Not({self.child!r})"


# --- Парсер запитів методом рекурсивного спуску ---

class QueryParser:
    """
    Парсер складних булевих запитів методом рекурсивного спуску.
    Граматика:
      expr   := or_expr
      or     := and_expr ('OR' and_expr)*
      and    := not_expr ('AND' not_expr)*
      not    := 'NOT' not_expr | atom
      atom   := '(' expr ')' | PHRASE | TERM
    """

    def __init__(self, text: str) -> None:
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> list[str]:
        token_pattern = re.compile(r'"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.IGNORECASE)
        tokens = token_pattern.findall(text)
        return tokens

    def _current(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def parse(self) -> Node:
        node = self._parse_or()
        return node

    def _parse_or(self) -> Node:
        node = self._parse_and()
        while self._current() and self._current().upper() == "OR":
            self.pos += 1
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> Node:
        node = self._parse_not()
        while self._current() and self._current().upper() not in ("OR", ")"):
            if self._current().upper() == "AND":
                self.pos += 1
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> Node:
        if self._current() and self._current().upper() == "NOT":
            self.pos += 1
            operand = self._parse_not()
            return Not(operand)
        return self._parse_atom()

    def _parse_atom(self) -> Node:
        token = self._current()
        if not token:
            raise ValueError("Неочікуваний кінець запиту")

        if token == "(":
            self.pos += 1
            sub_expr = self._parse_or()
            if self._current() == ")":
                self.pos += 1
            return sub_expr

        self.pos += 1

        if token.startswith('"') and token.endswith('"'):
            phrase_words = list(tokenize(token[1:-1]))
            return Phrase(phrase_words)

        return Term(token)


def parse_query(text: str) -> Node:
    """Зручна обгортка для парсингу рядка запиту в дерево AST."""
    return QueryParser(text).parse()