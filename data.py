from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator, Vocab


def get_train_data(
    seq_len: int,
) -> tuple[npt.NDArray[np.int32], Vocab, Callable[[str], list[str]]]:
    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter = WikiText2(split="train")
    lines_np = [np.array(vocab(tokenizer(item)), dtype=int) for item in train_iter]
    all_text_np = np.concatenate(lines_np)

    data_size = all_text_np.shape[0] // seq_len
    all_text_np = all_text_np[: data_size * seq_len]

    seqs = all_text_np.reshape(-1, seq_len)
    print(f"sequences shape: {seqs.shape}")
    return seqs, vocab, tokenizer
