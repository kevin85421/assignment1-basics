from pathlib import Path
from collections import defaultdict
import regex as re

def _get_new_word(word: tuple[bytes, ...], most_common_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i == len(word) - 1:
            new_word.append(word[i])
            break

        pair = (word[i], word[i+1])
        if pair == most_common_pair:
            new_word.append(most_common_pair[0] + most_common_pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def _in_place_reset_pair_freq_table(pair_freq_table: defaultdict[tuple[bytes, bytes], int], word: tuple[bytes, ...], freq: int) -> None:
    """
    In-place reset the pair frequency table for a word.

    Args:
        pair_freq_table: a dictionary of pair frequencies
        word: a word
        freq: the frequency of the word

    Returns:
        None
    """
    for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        pair_freq_table[pair] -= freq
        assert pair_freq_table[pair] >= 0
        if pair_freq_table[pair] == 0:
            del pair_freq_table[pair]

def _in_place_reset_pair_to_words(pair_to_words: defaultdict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]], word: tuple[bytes, ...]) -> None:
    """
    In-place reset the pair to words table for a word.

    Args:
        pair_to_words: a dictionary of pair to words
        word: a word

    Returns:
        None
    """
    for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        pair_to_words[pair][word] -= 1
        assert pair_to_words[pair][word] >= 0
        if pair_to_words[pair][word] == 0:
            del pair_to_words[pair][word]

def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = Path(input_path).read_text(encoding="utf-8")

    # Step 1: Vocabulary initialization
    vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Step 2: Pre-tokenization

    # Removing special tokens before pre-tokenization
    special_pat = re.compile("|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)))
    text_segments = re.split(special_pat, text)

    # TODO: parallelization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    freq_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
    for segment in text_segments:
        for m in re.finditer(PAT, segment):
            freq_table[tuple([bytes([b]) for b in m.group(0).encode("utf-8")])] += 1

    # key: pair, value: frequency of this pair
    pair_freq_table: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    # key: pair, value: a dictionary of words that contain this pair and how many times it appears in the word
    pair_to_words: defaultdict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]] = defaultdict(dict)
    for word in freq_table:
        for i in range(len(word) - 1):
            pair: tuple[bytes, bytes] = (word[i], word[i+1])
            pair_freq_table[pair] += freq_table[word]
            pair_to_words[pair][word] = pair_to_words[pair].get(word, 0) + 1

    # Step 3: Merging
    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        # Get the most common pair. If there are multiple pairs with the same frequency, choose the one with the greater lexical order.
        most_common_pair = max(pair_freq_table, key=lambda p: (pair_freq_table[p], p))
        merges.append(most_common_pair)

        most_common_pair_combined = most_common_pair[0] + most_common_pair[1]
        new_index = len(vocab)

        vocab[new_index] = most_common_pair_combined

        words = list(pair_to_words[most_common_pair].keys())
        new_words: list[tuple[bytes, ...]] = []
        for word in words:
            new_words.append(_get_new_word(word, most_common_pair))

        for word in words:
            freq_table_key = tuple([bytes([b]) for ele in word for b in ele])
            _in_place_reset_pair_freq_table(pair_freq_table, word, freq_table[freq_table_key])
            _in_place_reset_pair_to_words(pair_to_words, word)

        for word in new_words:
            freq_table_key = tuple([bytes([b]) for ele in word for b in ele])
            for i in range(len(word) - 1):
                pair: tuple[bytes, bytes] = (word[i], word[i+1])
                pair_freq_table[pair] += freq_table[freq_table_key]
                pair_to_words[pair][word] = pair_to_words[pair].get(word, 0) + 1

        assert most_common_pair not in pair_freq_table
        del pair_to_words[most_common_pair]

    return vocab, merges
