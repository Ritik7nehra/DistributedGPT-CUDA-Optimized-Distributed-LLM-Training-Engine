import torch
from dataset import TextDataset, load_dataset

def test_next_token_shift():
    data=torch.arange(20); ds=TextDataset(data,block_size=5); x,y=ds[0]
    assert torch.equal(x,data[0:5]); assert torch.equal(y,data[1:6])

def test_len_matches_formula():
    data=torch.arange(100); assert len(TextDataset(data,10))==90

def test_too_short_raises():
    try: TextDataset(torch.arange(5),10); assert False
    except ValueError: pass

def test_load_dataset_splits(tmp_path):
    p=tmp_path/"corpus.txt"; p.write_text("the quick brown fox "*50,encoding="utf-8")
    train,val,tok=load_dataset(str(p),block_size=16,train_split=.9)
    assert len(train)>0 and len(val)>0
    x,y=train[0]; assert x.shape==(16,) and y.shape==(16,); assert torch.equal(x[1:],y[:-1])
