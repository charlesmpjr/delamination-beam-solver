# Cheap route: classical lamination beam + spec's update rule (no FE at all)
import os,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
E1,E2,E3,G12,G13,G23,n12,n13,n23=171420.,9080.,9080.,5290.,5290.,3270.,.32,.32,.43
SI,KI,GI=80.,1e5,.277;S2,K2,G2=227.6,2.84e5,.788
th=np.radians([0,45,-45,90,90,-45,45,0])
def law(d,S,K,G):
    d0=S/K;df=2*G/S;return np.where(d>d0,df/(df-d0)*(1-d0/np.maximum(d,1e-300)),0).clip(0,1)
def mixed(a,b):
    g1=.5*KI*a*a;g2=.5*K2*b*b
    if g1+g2<=0:return 0.
    be=g2/(g1+g2);Gm=GI+(G2-GI)*be**1.75;dm=np.hypot(a,b);d1=SI/KI;d2=S2/K2
    d0=np.sqrt(d1*d1+(d2*d2-d1*d1)*be);S=np.sqrt(SI*SI+(S2*S2-SI*SI)*be);df=2*Gm/S
    return float(np.clip(df*(dm-d0)/(dm*(df-d0)),0,1)) if dm>d0 and df>d0 else 0.
def Ex():
    c,s=np.cos(th),np.sin(th);n21=n12*E2/E1
    Sx=c**4/E1+s**4/E2+(1/G12-2*n12/E1)*c*c*s*s
    Sxb=1/Sx
    m=1-n12*n21;Q11,Q22,Q12,Q66=E1/m,E2/m,n12*E2/m,G12
    Qb=Q11*c**4+2*(Q12+2*Q66)*c*c*s*s+Q22*s**4
    return Sxb,Qb
def sc(p,r):
    E=np.linalg.norm(p-r)/np.linalg.norm(r);return E,float(np.clip((1-E)/.95,0,1))
import sys;CAP=float(sys.argv[1]) if len(sys.argv)>1 else 1-1e-6
for nm,ex in zip(('1/S11 per ply','Qbar11'),Ex()):
  tot=[]
  for n,La in(('h8p',5),('h4p',5),('h8q',50)):
    r=np.load(HERE+'/data/'+n+'.npz');D,O,P,x,O2,h=r['damage'],r['opening'],r['levels'],r['x'],r['sliding'],float(r['h'])
    t=h/8;z=-h/2+t*np.arange(9);Db=np.sum(ex*(z[1:]**3-z[:-1]**3)/3)
    d=0.;Dp=np.zeros_like(D);Op=np.zeros_like(O);Sp=np.zeros_like(O)
    for i,p in enumerate(P):
        sl=2*t*(p/25*h)*x/Db
        op=np.where(x>=200-La,2*t*p/(25*La)/(E3*max(1-d,1e-6)**2),0)
        Op[i]=op;Sp[i]=sl;Dp[i]=np.maximum(law(op,SI,KI,GI),law(sl,S2,K2,G2))[None]
        dd=mixed(op.max(),sl.max());d=min(max(d,dd),CAP)
    a,b=sc(Dp,D),sc(Op,O);tot+=[a[1],b[1]]
    print(nm,n,'damage E/score %.3f %.3f'%a,'opening %.3f %.3f'%b,'sliding step1 max ref/pred',O2[0].max(),Sp[0].max())
  print(nm,'TOTAL',np.mean(tot))
