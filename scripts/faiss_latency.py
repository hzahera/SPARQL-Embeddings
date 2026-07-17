import time, numpy as np, faiss, sys
faiss.omp_set_num_threads(1)
rng = np.random.default_rng(0)
D = 128
def bench(N, nq, k=20):
    X = rng.standard_normal((N, D)).astype('float32'); faiss.normalize_L2(X)
    Q = X[rng.choice(N, nq, replace=False)].copy()
    t=time.time(); flat=faiss.IndexFlatIP(D); flat.add(X); bt_flat=time.time()-t
    t=time.time(); flat.search(Q,k); lat_flat=(time.time()-t)/nq*1e3
    t=time.time(); h=faiss.IndexHNSWFlat(D,32); h.hnsw.efConstruction=128; h.add(X); bt_h=time.time()-t
    h.hnsw.efSearch=64
    t=time.time(); h.search(Q,k); lat_h=(time.time()-t)/nq*1e3
    print(f"N={N:>9,} | FlatIP {lat_flat:.3f} ms/q ({1e3/lat_flat:,.0f} q/s) build {bt_flat:.2f}s "
          f"| HNSW {lat_h:.3f} ms/q ({1e3/lat_h:,.0f} q/s) build {bt_h:.1f}s | idx {N*D*4/1e6:.0f} MB", flush=True)
N=int(sys.argv[1]); nq=int(sys.argv[2]); bench(N,nq)
