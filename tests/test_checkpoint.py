import os, torch
from checkpoint_utils import load_checkpoint,save_checkpoint
from model import GPT,GPTConfig

def test_save_load_restores_weights(tmp_path):
    c=GPTConfig(vocab_size=32,block_size=16,n_layer=2,n_head=2,n_embd=32,dropout=0.0); m=GPT(c); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    _,loss=m(torch.randint(0,32,(2,16)),torch.randint(0,32,(2,16))); loss.backward(); opt.step()
    p=str(tmp_path/"ckpt.pt"); save_checkpoint(p,m,opt,step=7,model_config=c); saved={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        for q in m.parameters(): q.add_(1)
    assert load_checkpoint(p,m,opt)==7
    for k,v in m.state_dict().items(): assert torch.equal(v,saved[k])

def test_checkpoint_atomic(tmp_path):
    c=GPTConfig(vocab_size=16,block_size=8,n_layer=1,n_head=1,n_embd=16,dropout=0.0); m=GPT(c); opt=torch.optim.AdamW(m.parameters()); p=str(tmp_path/"ckpt.pt")
    save_checkpoint(p,m,opt,1,model_config=c); assert os.path.exists(p) and not os.path.exists(p+".tmp")
