import os,sys,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"src"))
from distributed.fsdp_train import enable_activation_checkpointing
from model import Block,GPT,GPTConfig

def test_activation_checkpointing_forward_backward():
    torch.manual_seed(7); c=GPTConfig(vocab_size=32,block_size=8,n_layer=2,n_head=2,n_embd=16,dropout=0.0); m=GPT(c); enable_activation_checkpointing(m); x=torch.randint(0,32,(2,8)); _,loss=m(x,x); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters() if p.requires_grad)

def test_checkpointing_wraps_blocks():
    c=GPTConfig(vocab_size=16,block_size=8,n_layer=2,n_head=1,n_embd=16,dropout=0.0); m=GPT(c); enable_activation_checkpointing(m)
    assert not any(isinstance(x,Block) for x in m.blocks)
