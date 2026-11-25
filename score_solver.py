"""Runs beam_solver.py on three beams and scores it against the stored results in data/.
Usage: python score_solver.py [as_coded|textbook]"""
import sys,os,time,platform,numpy as np
from beam_solver import simulate,MAT
H=os.path.dirname(os.path.abspath(__file__));mode=sys.argv[1] if len(sys.argv)>1 else 'as_coded'
print(platform.platform(),'python',platform.python_version(),'numpy',np.__version__,'mode',mode);tot=[]
for n in('h8p','h4p','h8q'):
    r=np.load(H+'/data/'+n+'.npz');t=time.time()
    d,o=simulate(MAT,[0,45,-45,90,90,-45,45,0],float(r['h']),200.,25.,40,int(r['nel']),r['levels'],mode);t=time.time()-t
    for nm,a,b in(('damage',d,r['damage']),('opening',o,r['opening'])):
        E=np.linalg.norm(a-b)/np.linalg.norm(b);s=float(np.clip((1-E)/.95,0,1));tot.append(s);print(n,nm,'E %.2e score %.3f'%(E,s))
    print(n,'time %.2f s'%t)
print('TOTAL %.3f'%np.mean(tot))
