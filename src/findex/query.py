import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    WORD = auto()
    PHRASE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass(slots=True)
class Token:
    type: TokenType
    value: str


@dataclass(slots=True)
class Term:
    value: str


@dataclass(slots=True)
class Phrase:
    terms: list[str]


@dataclass(slots=True)
class And:
    left: "Node"
    right: "Node"


@dataclass(slots=True)
class Or:
    left: "Node"
    right: "Node"


@dataclass(slots=True)
class Not:
    child: "Node"


Node = Term | Phrase | And | Or | Not


class QueryLexer:
    """Лексер для булевих пошукових запитів."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def tokenize(self) -> Iterator[Token]:
        phrase_regex = re.compile(r'"([^"]*)"')
        word_regex = re.compile(r"[\w]+(?:['’\-][\w]+)*")

        while self.pos < len(self.text):
            if self.text[self.pos].isspace():
                self.pos += 1
                continue

            if self.text[self.pos] == "(":
                self.pos += 1
                yield Token(TokenType.LPAREN, "(")
                continue

            if self.text[self.pos] == ")":
                self.pos += 1
                yield Token(TokenType.RPAREN, ")")
                continue

            if self.text[self.pos] == '"':
                match = phrase_regex.match(self.text, self.pos)
                if match:
                    self.pos = match.end()
                    yield Token(TokenType.PHRASE, match.group(1))
                    continue
                self.pos += 1
                continue

            match = word_regex.match(self.text, self.pos)
            if match:
                val = match.group(0)
                self.pos = match.end()
                val_upper = val.upper()
                if val_upper == "AND":
                    yield Token(TokenType.AND, val)
                elif val_upper == "OR":
                    yield Token(TokenType.OR, val)
                elif val_upper == "NOT":
                    yield Token(TokenType.NOT, val)
                else:
                    yield Token(TokenType.WORD, val.lower())
                continue

            self.pos += 1

        yield Token(TokenType.EOF, "")


class QueryParser:
    """Парсер для побудови AST дерева з булевих виразів."""

    def __init__(self, query: str) -> None:
        self.tokens = list(QueryLexer(query).tokenize())
        self.pos = 0

    @property
    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "")

    def advance(self) -> None:
        if self.pos < len(self.tokens):
            self.pos += 1

    def parse(self) -> Node:
        node = self.parse_or()
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.current_token.type == TokenType.OR:
            self.advance()
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.current_token.type in (
            TokenType.AND,
            TokenType.WORD,
            TokenType.PHRASE,
            TokenType.LPAREN,
            TokenType.NOT,
        ):
            if self.current_token.type == TokenType.AND:
                self.advance()
            right = self.parse_not()
            node = And(node, right)
        return node

    def parse_not(self) -> Node:
        if self.current_token.type == TokenType.NOT:
            self.advance()
            child = self.parse_primary()
            return Not(child)
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.current_token
        if tok.type == TokenType.WORD:
            self.advance()
            return Term(tok.value)
        if tok.type == TokenType.PHRASE:
            self.advance()
            words = [w.lower() for w in re.findall(r"[\w]+(?:['’\-][\w]+)*", tok.value)]
            return Phrase(words)
        if tok.type == TokenType.LPAREN:
            self.advance()
            node = self.parse_or()
            if self.current_token.type == TokenType.RPAREN:
                self.advance()
            return node
        self.advance()
        return Term("")
