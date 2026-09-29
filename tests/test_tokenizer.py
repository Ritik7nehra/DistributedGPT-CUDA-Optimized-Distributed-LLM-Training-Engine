import pytest
from tokenizer import CharTokenizer

def test_roundtrip():
    t="hello, distributed gpt!"; tok=CharTokenizer.from_text(t); assert tok.decode(tok.encode(t))==t

def test_vocab_size():
    assert CharTokenizer.from_text("aabbcc").vocab_size==3

def test_unknown_char():
    with pytest.raises(ValueError): CharTokenizer.from_text("abc").encode("xyz")

def test_save_load(tmp_path):
    tok=CharTokenizer.from_text("hello world"); p=str(tmp_path/"tok.json"); tok.save(p); loaded=CharTokenizer.load(p)
    assert loaded.chars==tok.chars and loaded.encode("hello")==tok.encode("hello")
