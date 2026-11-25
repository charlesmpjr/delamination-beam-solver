"""Global-local quasi-2D beam element with one-parameter damage, rewritten in Python/NumPy.
mode 'as_coded' keeps the conventions of the reference implementation;
mode 'textbook' uses the standard rotated stiffness and ply-by-ply interface factors."""
import sys,time,numpy as np
G=1/np.sqrt(3)
def law(d,S,K,Gc):
    d0=S/K;df=2*Gc/S;return np.where(d>d0,df/(df-d0)*(1-d0/np.maximum(d,1e-300)),0).clip(0,1)
def mixed(a,b,m):
    g1=.5*m['KI']*a*a;g2=.5*m['KII']*b*b
    if g1+g2<=0:return 0.
    be=g2/(g1+g2);Gm=m['GI']+(m['GII']-m['GI'])*be**m['eta'];dm=np.hypot(a,b);d1=m['SI']/m['KI'];d2=m['SII']/m['KII']
    d0=np.sqrt(d1*d1+(d2*d2-d1*d1)*be);S=np.sqrt(m['SI']**2+(m['SII']**2-m['SI']**2)*be);df=2*Gm/S
    return float(np.clip(df*(dm-d0)/(dm*(df-d0)),0,1)) if dm>d0 and df>d0 else 0.
def ply_stiffness(m,th,d,mode):
    E1,E2,E3=m['E1'],m['E2'],m['E3'];v12,v13,v23=m['v12'],m['v13'],m['v23'];v21=v12*E2/E1;v31=v13*E3/E1;v32=v23*E3/E2
    D=1-v12*v21-v13*v31-v23*v32-v12*v23*v31-v13*v21*v32;r=1-d
    c11=E1*(1-v23*v32)/D;c12=E1*(v21+v23*v31)/D;c13=E1*(v31+v21*v32)/D*r
    c22=E2*(1-v13*v31)/D;c23=E2*(v32+v12*v31)/D*r;c33=E3*(1-v12*v21)/D*r*r
    c44=m['G23'];c55=m['G13']*max(1e-6,r);c66=m['G12']
    c,s=np.cos(np.radians(th)),np.sin(np.radians(th))
    if abs(c)<1e-12:c=0.
    if abs(s)<1e-12:s=0.
    b11=c11*c**4+c22*s**4+2*s*s*c*c*(c12+2*c66);b12=(c11+c22)*s*s*c*c+c12*(c**4+s**4)-4*s*s*c*c*c66
    b22=c11*s**4+c22*c**4+2*s*s*c*c*(c12+2*c66);b13=c13*c*c+c23*s*s;b23=c13*s*s+c23*c*c
    a=c22 if mode=='as_coded' else c12
    b16=(c11-a)*c**3*s+(c12-c22)*c*s**3-2*c*s*(c*c-s*s)*c66
    b26=(c11-c12)*c*s**3+(c12-c22)*c**3*s+2*c*s*(c*c-s*s)*c66
    b36=0. if mode=='as_coded' else (c13-c23)*c*s
    b66=(c11+c22-2*c12)*c*c*s*s+c66*(c*c-s*s)**2
    b44=c44*c*c+c55*s*s;b45=(c55-c44)*c*s;b55=c44*s*s+c55*c*c
    A=np.array([[b11,b13],[b13,c33]]);B=np.array([[b12,b16],[b23,b36]]);Dm=np.array([[b22,b26],[b26,b66]])
    F=np.array([[b12,b23],[b16,b36]])
    return A-B@np.linalg.solve(Dm,F),b55-b45**2/b44
def solve_level(m,theta,h,L,ne,nel,P,b,d,mode):
    """returns opening and sliding, arrays (interfaces, 6*ne)"""
    d=min(d,1-1e-6);n=len(theta);N=2*n;t=h/n;Le=L/ne
    Q,q5=zip(*[ply_stiffness(m,x,d,mode) for x in theta]);Q=np.array(Q);q5=np.array(q5);q13=Q[:,0,1];q33=Q[:,1,1]
    g=lambda z:np.array([z,z*z]);gz=lambda z:np.array([1.,2*z])*2/t
    def rhs(q):
        v=np.zeros(N);v[0]=-q[0];v[-1]=-q[-1];v[1:-1:2]=q[1:]-q[:-1];return v
    def Cm(q):
        M=np.zeros((N,N));M[0,:2]=q[0]*gz(-1);M[-1,-2:]=q[-1]*gz(1)
        for k in range(n-1):
            f=(q[0]+q[1])/(2*t) if mode=='as_coded' else (q[k]+q[k+1])/(2*t)
            M[2*k+1,2*k:2*k+2]=q[k]*gz(1);M[2*k+1,2*k+2:2*k+4]=-q[k+1]*gz(-1)
            M[2*k+2,2*k:2*k+2]=f*g(1);M[2*k+2,2*k+2:2*k+4]=-f*g(-1)
        return np.linalg.inv(M)
    def Bm(q):
        M=np.zeros((N,N));M[0,:2]=-q[0]*g(-1);M[-1,-2:]=-q[-1]*g(1)
        for k in range(n-1):M[2*k+1,2*k:2*k+2]=-q[k]*g(1);M[2*k+1,2*k+2:2*k+4]=q[k+1]*g(-1)
        return M
    C3,C5=Cm(q33),Cm(q5);B13,B55=Bm(q13),Bm(q5);uz_,wx_=rhs(q5),rhs(q5);ux_,wz_=rhs(q13),rhs(q33)
    Sw_wz=C3@wz_;Sw_ux=C3@ux_;Sw_uzx=C3@B13@C5@uz_;Su_uz=C5@uz_;Su_wx=C5@wx_;Su_wzx=C5@B55@C3@wz_
    sz=P/(b*nel*Le);sv=np.zeros(N);sv[0]=sz;sv[-1]=sz;Sw1=C3@sv
    Lf=lambda Z:np.array([(1-Z)/2,(1+Z)/2]);Lz=np.array([-1,1])/h
    def sh(x):
        a,c=(1-x)/2,(1+x)/2;Nu=np.zeros((8,2));Nw=np.zeros((8,2));Nux=np.zeros((8,2));Nwx=np.zeros((8,2))
        for i in range(2):
            Nu[2*i,i]=a;Nu[4+2*i,i]=c;Nw[2*i+1,i]=a;Nw[5+2*i,i]=c
            Nux[2*i,i]=-1/Le;Nux[4+2*i,i]=1/Le;Nwx[2*i+1,i]=-1/Le;Nwx[5+2*i,i]=1/Le
        return Nu,Nw,Nux,Nwx
    Zp=lambda k,z:((k+.5)*2/n-1)+z/n
    gp4,gw4=np.polynomial.legendre.leggauss(4)
    Ke=np.zeros((8,8));Fe=np.zeros(8)
    for k in range(n):
        s2=slice(2*k,2*k+2)
        for z,w in zip(gp4,gw4):
            l=Lf(Zp(k,z));f,fz=g(z),gz(z)
            for x in(-G,G):
                Nu,Nw,Nux,Nwx=sh(x);W=w*t/2*Le/2
                ex=Nux@(l+(Su_uz[s2]@f)*Lz)
                ez=Nw@(Lz+(Sw_wz[s2]@fz)*Lz)+Nux@((Sw_ux[s2]@fz)*l+(Sw_uzx[s2]@fz)*Lz)
                ga=Nu@(Lz+(Su_uz[s2]@fz)*Lz)+Nwx@((Su_wx[s2]@fz)*l+(Su_wzx[s2]@fz)*Lz)+Nwx@(l+(Sw_wz[s2]@f)*Lz)
                B=np.stack([ex,ez]);Ke+=(B.T@Q[k]@B+q5[k]*np.outer(ga,ga))*W
                Fe-=B.T@Q[k]@np.array([0,Sw1[s2]@fz])*W
    for x in(-G,G):
        Nu,Nw,Nux,Nwx=sh(x)
        for sg,s2,z in((-1,slice(0,2),-1),(1,slice(N-2,N),1)):
            l=Lf(z);f=g(z);v=Nw@(l+(Sw_wz[s2]@f)*Lz)+Nux@((Sw_ux[s2]@f)*l+(Sw_uzx[s2]@f)*Lz);Fe+=sg*sz*v*Le/2
    nd=4*(ne+1);K=np.zeros((nd,nd));F=np.zeros((nd,2))
    for e in range(ne):
        K[4*e:4*e+8,4*e:4*e+8]+=Ke
        if e>=ne-nel:F[4*e:4*e+8,0]+=Fe
    F[-4,1]=-P/b;F[-2,1]=P/b
    u=np.zeros((nd,2));u[4:]=np.linalg.solve(K[4:,4:],F[4:])
    dl=np.zeros((2,n,6*ne))
    for j,x in enumerate(np.linspace(-1,1,6)):
        Nu,Nw,Nux,Nwx=sh(x)
        for k in range(n):
            s2=slice(2*k,2*k+2);vw=0;vu=0
            for sg,z in((1,1),(-1,-1)):
                l=Lf(Zp(k,z));f=g(z)
                vw=vw+sg*(Nw@(l+(Sw_wz[s2]@f)*Lz)+Nux@((Sw_ux[s2]@f)*l+(Sw_uzx[s2]@f)*Lz))
                vu=vu+sg*(Nu@(l+(Su_uz[s2]@f)*Lz)+Nwx@((Su_wx[s2]@f)*l+(Su_wzx[s2]@f)*Lz))
            for e in range(ne):
                dl[0,k,6*e+j]=abs(vw@u[4*e:4*e+8,0]);dl[1,k,6*e+j]=abs(vu@u[4*e:4*e+8,1])
    return dl[0,:-1]+dl[0,1:],dl[1,:-1]+dl[1,1:]
def simulate(m,theta,h,L,b,ne,nel,levels,mode):
    d=0.;O=[];D=[]
    for P in levels:
        a,s=solve_level(m,theta,h,L,ne,nel,P,b,d,mode)
        O.append(a[0]);D.append(np.maximum(law(a,m['SI'],m['KI'],m['GI']),law(s,m['SII'],m['KII'],m['GII'])))
        d=max(d,mixed(a.max(),s.max(),m))
    return np.array(D),np.array(O)
MAT=dict(E1=171420.,E2=9080.,E3=9080.,G12=5290.,G13=5290.,G23=3270.,v12=.32,v13=.32,v23=.43,SI=80.,KI=1e5,GI=.277,SII=227.5812274368231,KII=284476.5342960289,GII=.788,eta=1.75)
