import time, numpy as np, faiss, sys
rng = np.random.default_rng(0); D=128
N=int(sys.argv[1]); nq=500; k=20; hnsw=sys.argv[2]=='1'
X=rng.standard_normal((N,D)).astype('float32'); faiss.normalize_L2(X)
Q=X[rng.choice(N,nq,replace=False)].copy()
faiss.omp_set_num_threads(4)
flat=faiss.IndexFlatIP(D); flat.add(X)
faiss.omp_set_num_threads(1)                       # single-thread search latency
t=time.time(); flat.search(Q,k); lf=(time.time()-t)/nq*1e3
line=f"N={N:>9,} | FlatIP {lf:.3f} ms/q ({1e3/lf:,.0f} q/s) | idx {N*D*4/1e6:.0f} MB"
if hnsw:
    faiss.omp_set_num_threads(4)
    t=time.time(); h=faiss.IndexHNSWFlat(D,32); h.hnsw.efConstruction=128; h.add(X); bt=time.time()-t
    h.hnsw.efSearch=64; faiss.omp_set_num_threads(1)
    t=time.time(); h.search(Q,k); lh=(time.time()-t)/nq*1e3
    line+=f" | HNSW {lh:.3f} ms/q ({1e3/lh:,.0f} q/s) build {bt:.1f}s"
print(line, flush=True)
