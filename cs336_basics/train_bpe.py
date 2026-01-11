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

    freq_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
    for segment in text_segments:
        for m in re.finditer(PAT, segment):
            freq_table[tuple([bytes([b]) for b in m.group(0).encode("utf-8")])] += 1

    # text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    # freq_table = defaultdict(int)
    # for m in text.split():
    #     freq_table[tuple(m)] += 1
    # print(freq_table)
    # import pickle
    # p = Path("/root/xai/x/kaihsun/assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl")
    # snap = pickle.loads(p.read_bytes())
    # print("snapshot merges: ")
    # for i, m in enumerate(snap["merges"]):
    #     print(i, m)


    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pair_freq_table: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
        for word in freq_table:
            for i in range(len(word) - 1):
                pair: tuple[bytes, bytes] = (word[i], word[i+1])
                pair_freq_table[pair] += freq_table[word]

        # Get the most common pair. If there are multiple pairs with the same frequency, choose the one with the greater lexical order.
        most_common_pair = max(pair_freq_table, key=lambda p: (pair_freq_table[p], p))
        merges.append(most_common_pair)

        # if len(merges) == 1:
        #     print("pair_freq_table: ", pair_freq_table)
        #     print("  max pair: ", most_common_pair)
        #     print("  max pair freq: ", pair_freq_table[most_common_pair])

        most_common_pair_word = most_common_pair[0] + most_common_pair[1]
        new_index = len(vocab)

        # print("most common pair word: ", most_common_pair, "num_merges: ", len(merges))
        # print("  (e2, 80) pair freq table: ", pair_freq_table.get((0xe2, 0x80), 0))
        # print("  (o, n) pair freq table: ", pair_freq_table.get(("o", "n"), 0))
        # print("  (o, on) pair freq table: ", pair_freq_table.get(("o", "on"), 0))

        vocab[new_index] = most_common_pair_word
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
            # print("new_word: ", new_word, "type(new_word): ", type(new_word))
            freq_table[new_word] = freq

        # print("most common pair: ", most_common_pair)
        # print("freq_table: ", freq_table)
        # print("merges: ", merges)
    # print("number of merges: ", len(merges), "vocab size: ", len(vocab))
    return vocab, merges
