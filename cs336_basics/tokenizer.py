from collections.abc import Iterable, Iterator
from re import S
import regex as re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self._special_pat = re.compile("(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")")
        self._pretoken_pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    @classmethod
    def _apply_merges(cls, pretoken: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]:
        tokens = pretoken.encode("utf-8")
        while True:
            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                pair_combined = pair[0] + pair[1]
                if pair in merges:
                    new_tokens.append(pair_combined)
                    i += 2
                    break
                new_tokens.append(tokens[i])
                i += 1
            while i < len(tokens):
                new_tokens.append(tokens[i])
                i += 1
            if tokens == new_tokens:
                return [bytes([b]) for b in tokens]
            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        tokens = []
        for segment in re.split(self._special_pat, text):
            if segment in self.special_tokens:
                tokens.append(self._token_to_id[segment])
                continue
            for pretoken in re.finditer(self._pretoken_pat, segment):
                tokens.extend(self._apply_merges(pretoken.group(0), self.merges))

        return [self._token_to_id[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8")
