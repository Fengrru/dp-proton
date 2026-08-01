import numpy as np
from pw_afm import PWAFM
from exp_highdim import build_split_digits
def ms(a):
    a=np.asarray(a,float); return "%.4f+-%.4f"%(a.mean(),a.std())
def build(tasks,exc=None):
    a=PWAFM(T=0.3,sigma0=1.0,create_rule='dp',dp_alpha=1.0,base_scale=3.0,dp_diag_cov=True,readout='density',density_bw=1.0,n_sub=1)
    for X,y,_,_ in tasks:
        if exc is not None:
            m=y!=exc; X,y=X[m],y[m]
        if len(X): a.fit_stream(X,y)
    return a
def acc(a,X,y):
    p=a.classify(X); return {int(c):float((p[y==c]==c).mean()) for c in np.unique(y)}
SED=[0,1,2,3,4]
R={k:[] for k in ["dt0","dt1","dsh","n0","n1","rl","ro","it0","it1","io","ish"]}
for s in SED:
    tk,_=build_split_digits(n_comp=30,seed=s)
    X=np.vstack([t[2] for t in tk]); y=np.concatenate([t[3] for t in tk])
    a=build(tk); a0=acc(a,X,y); n0=float(a.novelty(X[y==9]).mean())
    a.delete_concept(9); a1=acc(a,X,y); n1=float(a.novelty(X[y==9]).mean())
    R["dt0"].append(a0[9]); R["dt1"].append(a1.get(9,0.0)); R["dsh"].append(max(abs(a1[c]-a0[c]) for c in a0 if c!=9)); R["n0"].append(n0); R["n1"].append(n1)
    a=build(tk); p0=a.classify(X)
    for b in a.basins:
        if b.label==7: b.label=99
    p1=a.classify(X); rm=np.where(p1==99,7,p1); R["rl"].append(float((rm==p0).mean())); mo=y!=7; R["ro"].append(float((p1[mo]==p0[mo]).mean()))
    a=build(tk,exc=9); a0=acc(a,X,y); p0=a.classify(X)
    Xtr=np.vstack([t[0] for t in tk]); ytr=np.concatenate([t[1] for t in tk]); Xt=Xtr[ytr==9]
    rng=np.random.default_rng(0); Xk=Xt[rng.choice(len(Xt),20,replace=False)]
    a.fit_stream(Xk,np.full(20,9)); a1=acc(a,X,y); p1=a.classify(X); mo=y!=9
    R["it0"].append(a0.get(9,0.0)); R["it1"].append(a1.get(9,0.0)); R["io"].append(float((p1[mo]==p0[mo]).mean())); R["ish"].append(max(abs(a1[c]-a0[c]) for c in a0 if c!=9))
    print("seed",s,"done",flush=True)
print("=== KNOWLEDGE EDITING digits30 5seeds ===")
print("DELETE  t9_acc",ms(R["dt0"]),"->",ms(R["dt1"]),"| others_shift",ms(R["dsh"]))
print("DELETE  novelty",ms(R["n0"]),"->",ms(R["n1"]))
print("RELABEL remap_identical",ms(R["rl"]),"| other_identical",ms(R["ro"]))
print("INJECT  t9_acc",ms(R["it0"]),"->",ms(R["it1"]),"| old_identical",ms(R["io"]),"| others_shift",ms(R["ish"]))
