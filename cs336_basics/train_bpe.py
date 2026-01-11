from pathlib import Path
from collections import defaultdict
import regex as re


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

    freq_table: defaultdict[list[str], int] = defaultdict(int)
    for segment in text_segments:
        for m in re.finditer(PAT, segment):
            freq_table[tuple(m.group(0))] += 1

    # text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    # freq_table = defaultdict(int)
    # for m in text.split():
    #     freq_table[tuple(m)] += 1
    # print(freq_table)

    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pair_freq_table: defaultdict[tuple[str, str], int] = defaultdict(int)
        for word in freq_table:
            for i in range(len(word) - 1):
                pair: tuple[str, str] = (word[i], word[i+1])
                pair_freq_table[pair] += freq_table[word]

        # Get the most common pair. If there are multiple pairs with the same frequency, choose the one with the greater lexical order.
        most_common_pair = max(pair_freq_table, key=lambda p: (pair_freq_table[p], p))
        merge = (most_common_pair[0].encode("utf-8"), most_common_pair[1].encode("utf-8"))
        merges.append(merge)

        new_index = len(vocab)
        most_common_pair_word = most_common_pair[0] + most_common_pair[1]
        vocab[new_index] = most_common_pair_word.encode("utf-8")
        outdated_words = {}
        

        for word in freq_table:
            new_word = []
            i = 0
            is_outdated = False
            while i < len(word):
                if i == len(word) - 1:
                    new_word.append(word[i])
                    break

                pair_word = word[i] + word[i+1]
                if pair_word == most_common_pair_word:
                    new_word.append(pair_word)
                    is_outdated = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if is_outdated:
                outdated_words[word] = new_word

        for outdated_word in outdated_words:
            freq = freq_table[outdated_word]
            del freq_table[outdated_word]
            new_word = tuple(outdated_words[outdated_word])
            freq_table[new_word] = freq

        # print("most common pair: ", most_common_pair)
        # print("freq_table: ", freq_table)
        # print("merges: ", merges)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    print(vocab)
    print(merges)