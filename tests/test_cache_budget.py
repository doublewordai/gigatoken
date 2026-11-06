"""The process-global cache-budget knob, end to end."""

import random

import gigatoken
from gigatoken.gigatoken_rs import BPETokenizer, SentencePieceTokenizer

DEFAULT = 512 * 1024 * 1024


def test_max_cache_bytes_knob(gpt2_tokenizer_path):
    assert gigatoken.get_max_cache_bytes() == DEFAULT
    # Enough distinct gibberish to overflow a 5 MiB budget
    # (gpt2's ~50k-entry seed plus ~80k new pretokens).
    rng = random.Random(1234)
    text = " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(6, 10)))
        for _ in range(80_000)
    )
    try:
        gigatoken.set_max_cache_bytes(None)
        assert gigatoken.get_max_cache_bytes() is None
        gigatoken.set_max_cache_bytes(5 * 1024 * 1024)
        assert gigatoken.get_max_cache_bytes() == 5 * 1024 * 1024
        small = BPETokenizer.from_hf(gpt2_tokenizer_path)
    finally:
        gigatoken.set_max_cache_bytes(DEFAULT)
    default = BPETokenizer.from_hf(gpt2_tokenizer_path)
    seed = small.cache_entries()
    assert seed == default.cache_entries()  # both start at vocab-seed level

    # The budget changes memory behavior only, never output: the default
    # cache keeps most distinct words, while the 5 MiB one wiped back
    # toward seed level along the way and stays well below.
    small_ids = small.encode(text)
    assert (small_ids == default.encode(text)).all()
    grown = default.cache_entries()
    assert grown > seed + 60_000
    assert small.cache_entries() < seed + 60_000
    assert small.cache_entries() < grown


def test_max_cache_bytes_sentencepiece(tinyllama_tokenizer_path):
    # SentencePiece caches start empty (no vocab seed); the budget covers
    # the growing maps past the fixed 24 MiB front cache, so 25 MiB leaves
    # ~1 MiB for ~80k distinct units.
    rng = random.Random(4321)
    text = " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(6, 10)))
        for _ in range(80_000)
    )
    try:
        gigatoken.set_max_cache_bytes(25 * 1024 * 1024)
        small = SentencePieceTokenizer.from_hf(tinyllama_tokenizer_path)
    finally:
        gigatoken.set_max_cache_bytes(DEFAULT)
    default = SentencePieceTokenizer.from_hf(tinyllama_tokenizer_path)
    assert small.cache_entries() == default.cache_entries() == 0

    small_ids = small.encode(text)
    assert (small_ids == default.encode(text)).all()
    grown = default.cache_entries()
    assert grown > 60_000
    assert small.cache_entries() < 60_000
    assert small.cache_entries() < grown
