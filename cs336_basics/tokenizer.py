from collections.abc import Iterable, Iterator
from re import Pattern
import regex as re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merge_to_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self._token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # If `special_tokens` is empty, `re.split` will split the text into segments of length 1. Therefore, we need to handle this case separately.
        self._special_pat: Pattern[str] | None = re.compile("(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")") if self.special_tokens else None
        self._pretoken_pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    @classmethod
    def _apply_merges(cls, pretoken: str, merge_to_rank: dict[tuple[bytes, bytes], int]) -> list[bytes]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
        while True:
            new_tokens: list[bytes] = []
            i = 0

            # In BPE, we need to find the pair with the lowest rank to merge first.
            # Rank is the index of the pair in the merges list.
            #
            # (pair, rank, index)
            min_rank_pair: tuple[tuple[bytes, bytes], int, int] | None = None
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                if pair in merge_to_rank:
                    rank = merge_to_rank[pair]
                    if min_rank_pair is None or rank < min_rank_pair[1]:
                        min_rank_pair = (pair, rank, i)
                i += 1

            if min_rank_pair is None:
                return tokens

            new_tokens.extend(tokens[:min_rank_pair[2]])
            new_tokens.append(min_rank_pair[0][0] + min_rank_pair[0][1])
            if min_rank_pair[2] + 2 < len(tokens):
                new_tokens.extend(tokens[min_rank_pair[2] + 2:])

            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        tokens = []
        segments: list[str] = re.split(self._special_pat, text) if self._special_pat else [text]
        for segment in segments:
            if segment in self.special_tokens:
                tokens.append(segment.encode("utf-8"))
                continue
            for pretoken in re.finditer(self._pretoken_pat, segment):
                tokens.extend(self._apply_merges(pretoken.group(0), self.merge_to_rank))

        return [self._token_to_id[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
