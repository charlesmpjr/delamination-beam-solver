"""Simple answers and bilinear-law check against the stored results in data/"""
import os,numpy as np
H=os.path.dirname(os.path.abspath(__file__));X={n:np.load(H+'/data/'+n+'.npz') for n in('h8p','h4p','h8q','h4q')}
def S(p,r):return float(np.clip((1-np.linalg.norm(p-r)/np.linalg.norm(r))/.95,0,1))
def law(d,S_,K,G):
    d0=S_/K;df=2*G/S_;return np.where(d>d0,df/(df-d0)*(1-d0/np.maximum(d,1e-300)),0).clip(0,1)
for n in('h8p','h4p','h8q'):
    D,O,P=X[n]['damage'],X[n]['opening'],X[n]['levels']
    print(n,'best constant: damage %.2f opening %.3f'%(max(S(np.full_like(D,c),D) for c in np.linspace(0,1,21)),max(S(np.full_like(O,c),O) for c in np.geomspace(O.max()/1e4,O.max(),40))))
    b=max((S(X[s]['damage'][j],D)+S(X[s]['opening'][j],O))/2 for s in X if s!=n for f in(.5,.75,1,1.5,2,3) for j in[np.clip(np.searchsorted(X[s]['levels']*f,P),0,len(X[s]['levels'])-1)])
    print(n,'rescaled other beam, total %.2f'%b)
    m=np.maximum(law(O,80.,1e5,.277),law(X[n]['sliding'],227.5812274368231,284476.5342960289,.788))
    print(n,'bilinear law vs stored damage, outer interfaces: %.1e %.1e'%(abs(m-D[:,0]).max(),abs(m-D[:,6]).max()),'levels',len(P),P[0],P[-1])
q=X['h4q'];print('h4q levels',len(q['levels']),'peak',q['levels'][-1],'last two max openings (um)',q['opening_max'][-2:]*1e3,'Mode II share',q['Gdiss_II'][-1]/(q['Gdiss_I'][-1]+q['Gdiss_II'][-1]))
