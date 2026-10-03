import os, sys, time, json
from collections import Counter
import numpy as np
from ldraw2solid import Library, Flattener, surfaces
from ldraw2solid.mesh import topology, tri_normals_area
lib = Library('lib'); fl = Flattener(lib)
names = sorted(f for f in os.listdir('lib/parts') if f.lower().endswith('.dat'))
t0=time.time(); err=Counter(); n=0; empty=0; wt=0; cert=0; miss=0
area=Counter(); cov=[]; notexact=Counter(); errs=[]
for nm in names:
    try:
        f = fl.flatten(nm)
        if len(f.tris)>400000: big=nm
    except Exception as e:
        err[type(e).__name__]+=1; errs.append((nm,str(e)[:80])); continue
    n+=1; fl.forget()
    if len(f.tris)==0: empty+=1; continue
    if f.missing: miss+=1
    if f.tri_certified.all(): cert+=1
    b,m,x = topology(f.tris)
    if b==0 and x==0: wt+=1
    _,a = tri_normals_area(f.tris)
    s = surfaces(f); tot=a.sum() or 1
    # area per instance
    ia = np.bincount(f.tri_instance, weights=a, minlength=len(f.instances))
    c=0
    for iid,sf in s.items():
        k = sf.kind if sf.exact else sf.kind+' ('+sf.why_not+')'
        area[k]+=ia[iid]/tot
        if sf.kind!='plane' or True: c+=ia[iid]/tot
    cov.append(c)
print('parts flattened',n,'errors',dict(err),'empty',empty,'time %.0fs'%(time.time()-t0))
print(errs[:10])
print('with missing subfiles',miss,' fully BFC-certified',cert,' already watertight after weld',wt)
cov=np.array(cov)
print('share of surface area coming from recognised primitives: mean %.0f%%, median %.0f%%'%(cov.mean()*100,np.median(cov)*100))
for q in (0.1,0.25,0.5,0.75): print('  parts with >=%d%% primitive area: %.0f%%'%(q*100,(cov>=q).mean()*100))
print({k:round(v/len(cov)*100,1) for k,v in area.most_common()})
