import torch
from model import GPT,GPTConfig

def cfg(**kw):
    d=dict(vocab_size=32,block_size=16,n_layer=2,n_head=2,n_embd=32,dropout=0.0); d.update(kw); return GPTConfig(**d)

def test_forward_shapes():
    c=cfg(); m=GPT(c); x=torch.randint(0,c.vocab_size,(4,c.block_size)); logits,loss=m(x,x)
    assert logits.shape==(4,c.block_size,c.vocab_size) and loss is not None and loss.item()>0

def test_forward_without_targets():
    c=cfg(); logits,loss=GPT(c)(torch.randint(0,c.vocab_size,(2,c.block_size)))
    assert loss is None and logits.shape==(2,c.block_size,c.vocab_size)

def test_causal_mask_blocks_future():
    c=cfg(); m=GPT(c).eval(); x=torch.randint(0,c.vocab_size,(1,c.block_size)); x2=x.clone(); x2[0,-1]=(x2[0,-1]+1)%c.vocab_size
    with torch.no_grad(): a,_=m(x); b,_=m(x2)
    assert torch.allclose(a[:,:-1],b[:,:-1],atol=1e-6)

def test_generate_extends_sequence():
    c=cfg(); m=GPT(c).eval(); x=torch.randint(0,c.vocab_size,(1,4)); out=m.generate(x,10)
    assert out.shape==(1,14) and torch.equal(out[:,:4],x)

def test_loss_decreases():
    torch.manual_seed(0); c=cfg(); m=GPT(c); opt=torch.optim.AdamW(m.parameters(),lr=1e-2); x=torch.randint(0,c.vocab_size,(8,c.block_size)); y=torch.randint(0,c.vocab_size,(8,c.block_size)); losses=[]
    for _ in range(50):
        opt.zero_grad(set_to_none=True); _,loss=m(x,y); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1]<losses[0]
