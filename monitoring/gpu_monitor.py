"""Phase 23: lightweight nvidia-smi GPU utilization/memory monitor."""
import argparse,shutil,subprocess,time
def poll_once():
    if not shutil.which("nvidia-smi"): return []
    q="index,utilization.gpu,memory.used,memory.total,temperature.gpu"
    out=subprocess.check_output(["nvidia-smi",f"--query-gpu={q}","--format=csv,noheader,nounits"],text=True,timeout=5)
    rows=[]
    for line in out.splitlines():
        i,u,mu,mt,temp=(x.strip() for x in line.split(","))
        rows.append({"index":int(i),"util_pct":float(u),"mem_used_mb":float(mu),"mem_total_mb":float(mt),"temp_c":float(temp)})
    return rows
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--interval",type=float,default=2); p.add_argument("--count",type=int,default=0); a=p.parse_args()
    if not shutil.which("nvidia-smi"): print("nvidia-smi not found"); raise SystemExit
    n=0
    while a.count==0 or n<a.count:
        for r in poll_once(): print(f"GPU{r['index']} util={r['util_pct']:.1f}% mem={r['mem_used_mb']:.0f}/{r['mem_total_mb']:.0f} MB temp={r['temp_c']:.0f}C")
        n+=1; time.sleep(a.interval)
