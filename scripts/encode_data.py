import pickle
import os
import pathlib
import sys
import re
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.adapters import Tokenizer
from scripts.utils import find_chunk_boundaries, compute_num_chunks
import numpy as np
from tqdm import tqdm

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_merges.pkl")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN_TXT_DATA_PATH = os.path.join(DATA_DIR, "owt_train.txt")
VAL_TXT_DATA_PATH = os.path.join(DATA_DIR, "owt_valid.txt")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "valid.dat")

special_tokens = ["<|endoftext|>"]

# Load vocab and merges
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
with open(MERGES_PATH, 'rb') as f:
    merges = pickle.load(f)

# Build tokenizer
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

print("=== Test Tokenizer ===")
test_texts = [
    "Once upon a time, there was a little robot.",
    "Hello world! <|endoftext|> Some more text.",
    "<|endoftext|>",
]

for text in test_texts:
    print(f"\nOriginal: {text}")
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded[:20], "..." if len(encoded) > 20 else "")
    decoded = tokenizer.decode(encoded)
    print("Roundtrip OK:", decoded == text)


def _encode_chunk(args):
    """Encode a file chunk and return token IDs as a numpy array."""
    filepath, start, end, vocab, merges, special_tokens, worker_id = args
    tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    with open(filepath, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")

    # Encode the full text (preserving newlines), show byte-level progress
    chunk_size = end - start
    pbar = tqdm(
        total=chunk_size, unit="B", unit_scale=True,
        desc=f"Worker {worker_id}", position=worker_id + 1,
        leave=False, mininterval=1.0,
    )
    # Process in sub-chunks split on special tokens for progress granularity
    special_pattern = (
        "(" + "|".join(map(re.escape, special_tokens)) + ")"
        if special_tokens else None
    )
    parts = re.split(special_pattern, text) if special_pattern else [text]
    all_ids = []
    for part in parts:
        all_ids.extend(tok.encode(part))
        pbar.update(len(part.encode("utf-8")))
    pbar.close()
    return np.array(all_ids, dtype=np.int32)


def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):
    file_size = os.path.getsize(path_to_txt)
    num_workers = mp.cpu_count()
    filepath = str(path_to_txt)
    split_token = special_tokens[0].encode("utf-8")
    num_chunks = compute_num_chunks(file_size, num_workers)

    if file_size > 1_000_000:
        boundaries = find_chunk_boundaries(filepath, num_chunks, split_token)
        chunk_args = [
            (filepath, s, e, tokenizer.vocab, tokenizer.merges, special_tokens, i % num_workers)
            for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]
        chunk_sizes = [e - s for s, e in zip(boundaries[:-1], boundaries[1:])]

        results = []
        ctx = mp.get_context("fork")
        pbar = tqdm(total=file_size, unit="B", unit_scale=True,
                    desc=f"Encoding {os.path.basename(path_to_txt)}")
        with ctx.Pool(num_workers) as pool:
            for i, token_arr in enumerate(pool.imap(_encode_chunk, chunk_args)):
                results.append(token_arr)
                pbar.update(chunk_sizes[i])
        pbar.close()
    else:
        # Small file: single process
        with open(path_to_txt, "r", encoding="utf-8") as f:
            text = f.read()
        results = [np.array(tokenizer.encode(text), dtype=np.int32)]

    # Concatenate and write memmap
    all_tokens = np.concatenate(results)
    total_tokens = len(all_tokens)
    print(f"  Total tokens: {total_tokens:,}")

    tokens_mm = np.memmap(save_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
    tokens_mm[:] = all_tokens
    tokens_mm.flush()
    print(f"  Saved to {save_path}")


def main():
    encode_txt_as_numpy_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    encode_txt_as_numpy_array(tokenizer, VAL_TXT_DATA_PATH, VAL_DATA_PATH)


if __name__ == "__main__":
    main()
