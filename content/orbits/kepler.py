# 03c0   π
# 025b   ɛ
# 20d7    ⃗

"α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω" 
"Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ   Σ Τ Υ Φ Χ Ψ Ω"

import numpy as np
import matplotlib.pyplot as plt

from numpy import cross, sin, cos, arctan2, pi, sqrt, arccos

π = pi

from numpy.linalg import norm
from scipy.optimize import fsolve

def vec(*x):
    return np.array(x)

def dist(u,v=0):
    return norm(np.array(u)-v)

def unit(v):
    return v / norm(v)


def RaDec2UnitV(ra,dec):
    '''Convert right ascension, declination (degrees) to unit vector'''
    ra,dec = np.radians((ra,dec))
    return vec(cos(dec) * cos(ra),
               cos(dec) * sin(ra),
               sin(dec)          )


def period(a,mu):
    '''Orbit period given major semiaxis'''
    return 2*π*sqrt(a**3/mu)


def anomV2E(θ,e):
    r = sqrt((1+e)/(1-e))
    E = 2*arctan2(sin(θ/2), r * cos(θ/2))
    return E

def anomE2V(E,e):
    r = sqrt((1+e)/(1-e))
    θ = 2*arctan2(r*sin(E/2), cos(E/2))
    return θ

def anomM2E(M,e):
    E = fsolve(lambda E: E - e*sin(E) - M, M)[0]
    return E


def Rotation2EulerAngles(R):
    X,Y,Z = R
    
    N = cross(np.array([0,0,1]),Z)
    if norm(N) < 1e-10:
        N = np.array([1,0,0])
    n = norm(N)
    
    i = np.arccos(Z[2])
    
    Ω = np.arccos(N[0]/n)
    if N[1] < 0:
        Ω = 2*π - Ω
    
    ω = arccos( N @ X / n)
    if X[2] < 0:
        ω = 2*π - ω
    
    return ω, Ω, i



# def EulerAnglesToRotation(ω, Ω, i):
#    return (rotation([0,0,1],Ω) @ rotation([1,0,0],i) @ rotation([0,0,1],ω)).T

# Para evitar dependencias de OpenCV sustituimos la rotación de eje arbitrario usando Rodrigues
# por rotaciones en los tres ejes, con el mismo convenio (lo que implica cambiar rot2 la posición
# del signo en rot2, que dejamos por completitud pero no se usa en los ángulos de Euler).

def rot3(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([[c, -s, 0]
                    ,[s,  c, 0]
                    ,[0,  0, 1]])

def rot1(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([[1, 0,  0]
                    ,[0, c, -s]
                    ,[0, s,  c]])

def rot2(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([[ c, 0, s]
                    ,[ 0, 1, 0]
                    ,[-s, 0, c]])


def EulerAnglesToRotation(ω, Ω, i):
    return (rot3(Ω) @ rot1(i) @ rot3(ω)).T





def odograph(h⃗, e⃗, μ):
    p = (h⃗ @ h⃗)/μ

    def v⃗(r⃗):
        return cross( h⃗/p , unit(r⃗)+e⃗ )
    
    return v⃗


def state2constants(r⃗,v⃗,μ):
    "compute major semiaxis, eccentricy, perifocal frame, and period"
    h⃗ = cross(r⃗,v⃗)
    e⃗ = cross(v⃗,h⃗)/μ - unit(r⃗)
    
    r = norm(r⃗)
    h = norm(h⃗)
    e = norm(e⃗)
    
    ε = (v⃗@v⃗)/2 - μ/r
    a = -μ/(2*ε)
    T = period(a,μ)
    
    Z = h⃗ / h
    X = e⃗ / e
    Y = cross(Z,X)

    return a, e, [X,Y,Z], T


def location(r⃗, orbit):
    a,e,[X,Y,Z],T = orbit

    θ = np.mod(arctan2(r⃗@Y , r⃗@X) , 2*π)
    E  = anomV2E(θ,e)
    M = E - e*sin(E)
    t = M/(2*π/T)
    
    return [θ,E,M], t


def keplerElements(r⃗,v⃗,μ):
    a, e, perifocal, _ = orbit = state2constants(r⃗,v⃗,μ)
    (_,_,M), _ = location(r⃗, orbit)
    ω, Ω, i = Rotation2EulerAngles(perifocal)
    return a, e, ω, Ω, i, M


def physconst(orbit):
    a,e,_,T = orbit
    μ = 4*π**2*a**3/T**2
    p = a*(1-e**2)
    h = sqrt(μ*p)
    ε = -μ/(2*a)
    return ε,h,μ


def mkPath(orbit):
    _,h,μ = physconst(orbit)
    a,e,axes,T = orbit
    
    [X,_,Z] = axes
    
    h⃗ = h*Z
    e⃗ = e*X
    vel = odograph(h⃗,e⃗,μ)
    
    def state(t):
        M = 2*π*t/T
        E = anomM2E(M,e)
        θ = anomE2V(E,e)
        r = a*(1-e*cos(E))
        r⃗ = r * vec(cos(θ),sin(θ),0)
        r⃗ = r⃗ @ axes
        v⃗ = odograph(h⃗, e⃗, μ)
        return r⃗, vel(r⃗)
    
    return state


def kepler2path(a, e, ω, Ω, i, T=1):
    axes = EulerAnglesToRotation(ω, Ω, i)
    orbit = a,e,axes,T
    return mkPath(orbit)


def mkPathAbsTime(orbit, p_r, t_r):
    t_nat = location(p_r,orbit)[-1]
    path = mkPath(orbit)
    def state(t):
        return path(t - t_r + t_nat)
    return state


#################################################################

# locus of second focus given two points in ellipse (first focus at origin)
def paramfocus(P,Q):
    c = dist(P,Q)/2
    a = (dist(P,(0,0))-dist(Q,(0,0)))/2
    #e = c/a
    b = sqrt(c**2-a**2)
    ang = arctan2(Q[1]-P[1],Q[0]-P[0])
    C = cos(ang)
    S = sin(ang)
    Z = (P[0]+Q[0])/2 , (P[1]+Q[1])/2
    def hyp(t):
        x = -a*np.cosh(t)
        y = b*np.sinh(t)
        X = C*x - S*y + Z[0]
        Y = S*x + C*y + Z[1]
        return X,Y
    return hyp

# parametric equation of ellipse given foci and a
def elipfocus(P,Q,a):
    c = dist(P,Q)/2
    e = c/a
    b2 = a**2-c**2
    if b2 < 0:
        return None
    b = np.sqrt(b2)
    ang = np.arctan2(Q[1]-P[1],Q[0]-P[0])
    C = np.cos(ang)
    S = np.sin(ang)
    Z = (P[0]+Q[0])/2 , (P[1]+Q[1])/2
    def elip(t):
        x = a*np.cos(t)
        y = b*np.sin(t)
        X = C*x - S*y + Z[0]
        Y = S*x + C*y + Z[1]
        return X,Y
    return elip


def asymptotes(p,q,k):
    p = np.array(p)
    q = np.array(q)
    a = k/2    
    c = dist(p,q)/2
    b = sqrt(c**2-a**2)
    O = (p+q)/2
    u = u1,u2 = unit(q-p)
    v = np.array([-u2,u1])
    A1 = O + u + b/a*v
    A2 = O + u - b/a*v
    l1 = cross( np.array([*O,1]) , np.array([*A1,1]) )
    l2 = cross( np.array([*O,1]) , np.array([*A2,1]) )
    return l1,l2


def shline(l,xmin=-10,xmax=10, **args):
    a,b,c = l / norm(l)
    if abs(b) < 1e-6:
        x = -c/a
        r = np.array([[x,xmin],[x,xmax]])
    else:
        y0 = (-a*xmin - c) / b
        y1 = (-a*xmax - c) / b
    plt.plot([xmin,xmax],[y0,y1], **args)


def quality(Ps,f):
    k = dist(Ps[0]) + dist(Ps[0],f)
    d = sum([ abs(dist(p,f) + dist(p) - k)/k for p in Ps ])
    return d


def Solve3Points(Ps):
    return InverseProjection2D(*Ps)[:2]


from scipy.optimize import minimize


# La intersección bruta de las hipérbolas necesita inicialización propicia.
# Se puede conseguir fácilmente intersectando las asimptotas. Esa inicialización
# casi puede valer como solución cuando los puntos están próximos
def Solve3PointsExperiment(Ps, show=False, newfig=True):

    return InverseProjection2D(*Ps)[:2]

    p1,p2,p3 = Ps
    H1 = paramfocus(p1,p2)
    H2 = paramfocus(p2,p3)

    t = fsolve(lambda v: np.array(H1(v[0]))-H2(v[1]),[1,-1])[0]
    f1 = 0,0
    f2 = H1(t)
    a = (dist(f1,p1) + dist(f2,p1))/2
    
    # otra idea: intersección de asimptotas
    print('Asymp')
    l1,l2 = asymptotes(p1,p2, abs(dist(p1)-dist(p2)) )
    print(l1)
    print(l2)
    m1,m2 = asymptotes(p3,p2, abs(dist(p2)-dist(p3)) )
    print(m1)
    print(m2)
    candi = [ cross(l,m) for l in [l1,l2] for m in [m1,m2] ]
    print(candi)
    candi = [ c[:2]/c[2] for c in candi ]
    print(candi)



    if show:
        if newfig: plt.figure(figsize=(8,8))
        plt.axis('equal')
        E = elipfocus(f1,f2,a)
        if E is not None:
            plt.plot(*E(np.linspace(0,2*np.pi,50)),color='red',ls='dashed',alpha=0.75)
            phi = np.linspace(-3,5,20)
            plt.plot(*H1(phi),color='green',lw=0.5)
            plt.plot(*H2(phi),color='green',lw=0.5)
            plt.plot(*f2,'o', color='blue')
            
            for p in Ps: plt.plot([f1[0],p[0]],[f1[1],p[1]],color='gray',lw=1, alpha=0.5)
            plt.plot(*f1,'.', color='orange', markerSize=15)
            plt.plot(*Ps.T,'.',color='black', markerSize=11)

            for l in [l1,l2,m1,m2]:
                shline(l,color='gray',lw=0.5)
            for x,y in candi:
                plt.plot(x,y,'.', color='brown')


    print('Q optim', quality(Ps,f2))
    f2_a = candi[0]
    a_a  = (dist(f2_a,p1) + dist(p1))/2
    print('Q asymp', quality(Ps,f2_a))
            
    # La optimización del foco va mejor así, con el default method pierde precisión
    #f2_opt = minimize(lambda v: quality(Ps,v),f2_a,method='Nelder-Mead')
    #print(f2_opt)

    return f2,a
    return f2_a,a_a


# oriented angle between two vectors
def angle(V1,V2):
    x1,y1 = V1
    x2,y2 = V2
    dot = x1*x2 + y1*y2      # dot product
    det = x1*y2 - y1*x2      # determinant
    return np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)


# (positive) difference between two angles (two solutions, ordered)
def difs(a1,a2):
    d1 = np.mod(a2-a1, 2*np.pi)
    d2 = 2*np.pi - d1
    if d2 < d1:
        return d2, d1
    else:
        return d1, d2


# create function to compute the eccentric anomaly of point 
# with respect to ellipse defined by foci and a
def eccenFromElip(P,Q,a):
    c = dist(P,Q)/2
    b = np.sqrt(a**2-c**2)
    e = c/a
    rho = np.sqrt((1+e)/(1-e))
    
    Z = (P[0]+Q[0])/2 , (P[1]+Q[1])/2
    D = P[0]-Q[0],P[1]-Q[1]
    
    def fun(X):
        u = X[0]-P[0], X[1]-P[1]
        trueAnom = V = angle( D, u )
        V = np.mod(V,2*np.pi)
        eccenAnom = E = 2*np.arctan2(np.sin(V/2), rho * np.cos(V/2))
        E = np.mod(E,2*np.pi)
        meanAnom  = M = E - e*np.sin(E)
        return V,E,M
    return fun

def deltaTime(p1,p2,mu,f2,a, debug=True):
    anom = eccenFromElip((0,0),f2,a)
    _,_,M1 = anom(p1)
    _,_,M2 = anom(p2)
    T = np.sqrt(4*np.pi**2*a**3/mu)
    n = 2*np.pi/T
    d1, d2 = difs(M1,M2)/n
    if debug:
        print('a={:.3f}, e={:.5f}'.format(a,np.linalg.norm(f2)/2/a))
        print('M1={:.3f}, M2={:.3f}'.format(M1,M2))
        print('t1={:.2f}, t2={:.2f}, dif1={:.3f}, dif2={:.3f}'.format(M1/n, M2/n, d1,d2))
    return difs(M1,M2)/n

def toPlane(r1,r2):
    eX = unit(r1)
    eZ = unit(np.cross(r1,r2))
    eY = unit(np.cross(eZ,eX))
    return np.array([eX,eY,eZ])


def ellipseFrom3Points(Rs, show=False, newfig=True):
    r1,r2,r3 = Rs
    Rot = toPlane(r1,r3)
    Ps = (Rs@Rot.T)
    #print(Ps)
    res = Ps[1,2]/dist(r1,r3)
    #print(res)
    Ps = Ps[:,:2]
    if show:
        f2,a = Solve3PointsExperiment(Ps, show=show, newfig=newfig)
    else:
        f2,a = Solve3Points(Ps)
    return f2,a,Rot


def orbitFromThreePoints(Rs,sign=1):
    f2,a,Rot = ellipseFrom3Points(Rs)
    c = norm(f2)/2
    e = c/a
    X = -unit(list(f2)+[0])
    Z = vec(0,0,np.sign(sign))
    Y = cross(Z,X)
    return a,e, [X,Y,Z]@Rot


def timesFromThreePoints(Rs,μ,show=False):
    f2,a,Rot = ellipseFrom3Points(Rs)
    p1,p2,p3 = Ps = (Rs@Rot.T)[:,:2]
    t_12 = deltaTime(p1,p2,μ,f2,a, debug=False)[0]
    t_23 = deltaTime(p2,p3,μ,f2,a, debug=False)[0]
    t_13 = deltaTime(p1,p3,μ,f2,a, debug=show)[0]
    return t_12, t_23, t_13, a, norm(f2)/2


def SolveAreas(Rs,Ds,Ts,rho_2):
    R_1,R_2,R_3 = Rs
    d_1,d_2,d_3 = Ds
    t_1,t_2,t_3 = Ts
    tau = (t_2-t_1)/(t_3-t_2)
    B3 = cross(R_1,R_2) -tau*cross(R_2,R_3) + rho_2*(cross(R_1,d_2)-tau*cross(d_2,R_3))
    A1 = -cross(d_1,R_2) - rho_2*cross(d_1,d_2)
    A2 = tau*cross(R_2,d_3) + tau*rho_2*cross(d_2,d_3)
    B = B3[[0,1]]
    A = np.vstack([A1,A2]).T
    rho_1, rho_3 = np.linalg.lstsq(A,B3,rcond=None)[0]
    return R_1 + rho_1*d_1, R_2 + rho_2*d_2, R_3 + rho_3*d_3


def SolveOnlyOne(Rs,Ds,rho_1,rho_3):
    R_1,R_2,R_3 = Rs
    d_1,d_2,d_3 = Ds
    
    r1,r3 = R_1 + rho_1*d_1, R_3 + rho_3*d_3
    hu = unit(cross(r1,r3))
    
    rho_2 = -(hu @ R_2)/(hu @ d_2)
    
    return r1, R_2 + rho_2*d_2, r3

################################################################

def mk_deltaTime(p1,p2,mu):
    H = paramfocus(p1,p2)
    def fun(t,debug=False):
        f1 = (0,0)
        f2 = H(t)
        a = (dist(f1,p1) + dist(f2,p1))/2
        return deltaTime(p1,p2,mu,f2,a,debug=debug)
    return fun

from scipy.optimize import minimize,fsolve

def Lambert2D(p1,p2,mu, dT, long=False):
    H = paramfocus(p1,p2)
    f1 = (0,0)
    D1 = dist(p1,f1)
    D2 = dist(p2,f1)
    def A(t):
        f2 = H(t)
        a = (dist(f1,p1) + dist(f2,p1))/2
        c = dist(f1,f2)/2
        #b2 = a**2-c**2
        return a, c/a
    
    f = mk_deltaTime(p1,p2,mu)
    
    
    if long:
        t = fsolve(lambda x: f(x[0])[1]-dT , 1) [0]
    else:
        t = fsolve(lambda x: f(x[0])[0]-dT , 1) [0]
    
    print('Lambert Error:', f(t), dT)
    
    a,e = A(t)
    energy = -mu/2/a
    mh = np.sqrt(mu**2*(e**2-1)/2/energy)
    vh = np.array([0,0,mh]) # ojo
    f2 = H(t)
    return f2,a

def Lambert(r1,r2,mu, dT, long=False):
    Rot = toPlane(r1,r2)
    p1,p2 = Ps = (np.array([r1,r2])@Rot.T)[:,:2]
    res = Lambert2D(p1,p2,mu,dT=dT,long=long)
    if dT is not None:
        f2, a = res
    else:
        return None
    c = norm(f2)/2
    e = c/a
    X = -unit(list(f2)+[0])
    Z = vec(0,0,np.sign(1))
    Y = cross(Z,X)
    return a,e, [X,Y,Z]@Rot, period(a, mu)

################################################################


def null1(M):
    u,s,vt = np.linalg.svd(M)
    return vt[-1,:]

def InverseProjection2D(p1,p2,p3):
    P = p1,p2,p3
    R = [ norm(p) for p in [p1,p2,p3] ]
    mat = np.array([ [*p,r,1] for p,r in zip(P,R) ])
    #print(mat)
    Π = null1(mat)
    #print(Π)
    p = -Π[3]/Π[2]
    print(f'p ~ {p:.4f}')
    
    g1 = p/norm(p1) - 1
    g3 = p/norm(p3) - 1

    #print(g1,g3)

    Δν = np.arccos( p1 @ p3 / norm(p1)/norm(p3) )
    print(f'Δν ~ {np.degrees(Δν):.2f}º')

    ν1 = np.arctan( -(g3/g1-cos(Δν)) / sin(Δν) )
    e = g1/cos(ν1)

    # no hace falta pero mejor así
    if e < 0:
        e  = abs(e)
        ν1 = ν1 + np.pi

    print(f'ν1 ~ {np.degrees(ν1):.2f}º')
    print(f'e ~ {e:.4f}')

    a = p/(1-e**2)
    print(f'a ~ {a:.4f}')

    u = vec(cos(ν1), sin(-ν1))
    f2 = -u * 2*a*e

    #print((p/(1+e*cos(ν1))), norm(p1))
    #print((p/(1+e*cos(ν1+Δν))), norm(p3))

    return f2,a,e


def Orbit3Points(r1,r2,r3,mu):
    Rot = toPlane(r1,r3)
    p1,p2,p3 = Ps = (np.array([r1,r2,r3])@Rot.T)[:,:2]
    f2,a,e = InverseProjection2D(p1,p2,p3)
    X = -unit(list(f2)+[0])
    Z = vec(0,0,np.sign(1))
    Y = cross(Z,X)
    return a,e, [X,Y,Z]@Rot, period(a, mu)

################################################################

def Gauss0(Rs,Ds,Ts, r2=None):
    R_1,R_2,R_3 = Rs
    d_1,d_2,d_3 = Ds
    t_1,t_2,t_3 = Ts
    tau12 = (t_2-t_1)/(t_3-t_1)
    tau23 = (t_3-t_2)/(t_3-t_1)
    G = 1
    if r2 is not None:
        Period = 365.25
        #Period = 364.8338413609741
        #Period = 365.286762497922
        G = 1 + (2*np.pi**2*(t_2-t_1)*(t_3-t_2)/Period**2/r2**3)    
    tau12 = tau12*G
    tau23 = tau23*G
    print(f'G={G:.4f}')
    F = R_1*tau23 + R_3*tau12
    n = unit(cross(d_1,d_3))
    Q3 = n
    Q4 = -n @ F
    Q = vec(*Q3,Q4)
    alpha_d = -Q3 @ d_2
    alpha_n =  Q3 @ R_2 + Q4
    alpha   = alpha_n / alpha_d
    P_2 = R_2 + alpha*d_2
    #print( Q@np.array([*F,1]) )
    print(f'τ_12={tau12:.3f}  τ_23={tau23:.3f}')
    print(f'F={np.round(F,1)}')
    print(f'n={np.round(n,3)}')
    print(f'Q={np.round(Q,3)}')
    print(f'α_n={alpha_n:.3f}  α_d={alpha_d:.3f}')
    print(f'α={alpha:.3e}')
    print(f'P_2={np.round(P_2,3)}')
    print(f'r2={np.round(norm(P_2),3)}')
    return F,P_2,G

################################################################

# Los puntos quedan muy bien aproximados, pero el factor G, aunque mejora mucho,
# deja un poco de error en el half-parameter.

# Con la proyección inversa sale clavado.

# Finalmente necesitamos la rotación general

def rodrigues(v,a):
    x,y,z = unit(v)
    K = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]])
    R = np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K @ K)
    return R


def myrotation(v, a):
    R = rodrigues(v,a)
    return R

def GaussOrbit(P,T):
    p1,p2,p3 = P
    t1,t2,t3 = T
    Period = 365
    r2 = norm(p2)
    G = 1 + (2*np.pi**2*(t2-t1)*(t3-t2)/Period**2/r2**3)  
    T13 = 0.5*norm(cross(p1,p3))
    S13 = T13*G
    Δt = t3-t1
    p = (S13/np.pi/(Δt/365))**2
    h = 2 * S13 / Δt

    print(f'p ~ {p:.4f}')
    print(f'h ~ {h:.4f}')

    g1 = p/norm(p1) - 1
    g3 = p/norm(p3) - 1
    Δν = np.arccos( p1 @ p3 / norm(p1)/norm(p3) )
    ν1 = np.arctan( -(g3/g1-cos(Δν)) / sin(Δν) )
    print(f'ν1 ~ {np.degrees(ν1):.2f}º')

    e = g1/cos(ν1)
    print(f'e ~ {e:.4f}')

    a = p/(1-e**2)
    print(f'a ~ {a:.4f}')

    Z = unit(cross(p1,p3))
    X = myrotation(Z, -ν1) @ unit(p1)
    Y = cross(Z,X)

    return a, e, [X,Y,Z], a**(3/2)*Period
    
################################################################

def Optimize3Points(P,E,D,T,mu):
    t1,t2,t3 = T
    depth1,depth3 = norm(P[0]-E[0]), norm(P[2]-E[2])
    sol,info,code,msg = fsolve(lambda x: timesFromThreePoints(SolveOnlyOne(E,D, x[0],x[1]), mu, show=False)[:2] - np.array([t2-t1, t3-t2]), (depth1,depth3), full_output=True)
    opt_rho1, opt_rho3 = sol
    print(msg)
    assert code == 1
    rP = SolveOnlyOne(E,D, opt_rho1, opt_rho3)
    return rP


def Optimize1Point(P,E,D,T,mu):
    t1,t2,t3 = T
    depth2 = norm(P[1]-E[1])
    sol,info,code,msg = fsolve(lambda x: timesFromThreePoints(SolveAreas(E,D, T, x), mu, show=False)[2] - (t3-t1), 1, full_output=True)
    opt_rho2 = sol
    print(msg)
    assert code == 1
    rP = SolveAreas(E,D, opt_rho2)
    return rP



def OptimizeBrute(P,E,D,T,mu,peri):
    t1,t2,t3 = T
    depth1,depth2,depth3 = norm(P[0]-E[0]), norm(P[1]-E[1]), norm(P[2]-E[2])
    def fun(x):
        t_12, t_23, t_13, a, c = timesFromThreePoints([ E[0]+x[0]*D[0],E[1]+x[1]*D[1],E[2]+x[2]*D[2] ], mu, show=False)
        return t_12 - (t2-t1), t_23 - (t3-t2), period(a,mu) - peri
    sol,info,code,msg = fsolve(fun, (depth1,depth2,depth3), full_output=True)
    opt_rho1, opt_rho2, opt_rho3 = sol
    print(msg)
    assert code == 1
    rP = [ E[0]+opt_rho1*D[0], E[1]+opt_rho2*D[1] ,E[2]+opt_rho3*D[2] ]
    return rP


################################################################


def info_orbit(orb):
    a,e,axis,T = orb
    p = a*(1-e**2)
    print(f'a={a:.4f}, e={e:.4f}, p={p:.4f}')
    print('ω={}, Ω={}, i={}'.format(*np.round(np.degrees(Rotation2EulerAngles(axis)),3)))
    ε,h,μ = physconst(orb)
    print(f'T={T/365:.2f}a, ε={ε:.4f}, h={h:.4f}, p={h**2/μ:.5f}')
    
################################################################

def DEC(g,m,s):
    return np.radians(g+m/60+s/3600)

def RA(h,m,s):
    return np.radians(15*(h+m/60+s/3600))   
    
def Sun_distance(d): #distancia al sol en UA, d es el dia del año
    s = 1.0 - 0.017*np.cos(0.986*d*np.pi/180-4*np.pi/180)
    return s

def DIR(ra,dec):
    px = np.cos(dec)*np.cos(ra)
    py = np.cos(dec)*np.sin(ra)
    pz = np.sin(dec)
    return np.array([px,py,pz])

################################################################

def prettyAngle(ang):
    v = np.degrees(ang)
    dg = int(v)
    v = (v - dg)*60
    dm = int(v)
    ds = (v-dm)*60
    return f'''{dg:.0f}º {dm:.0f}' {ds:.2f}"'''

################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def fullOrbit(ax, orb, name):
    fun = mkPath(orb)
    elip = np.array([fun(t)[0] for t in np.linspace(0,orb[-1],300)])
    ax.plot(*elip.T,color='#1f77b4',label=name)
    
    a,e,axes, _ = orb
    
    p = a*(1-e**2)
    
    ω,Ω,_ = Rotation2EulerAngles(axes)
    rOM = p/(1+e*cos(ω))
    rOM2 = p/(1+e*cos(ω+pi))
    
    xan, yan, zan = ascending_node = vec(rOM*cos(Ω),rOM*sin(Ω),0)
    xan2, yan2, zan2 = descending_node = vec(rOM2*cos(Ω+pi),rOM2*sin(Ω+pi),0)
    cencir = (ascending_node+descending_node)/2
    radcir = dist(ascending_node,descending_node)/2
    
    circle = cencir+radcir*np.array([(cos(t),sin(t),0) for t in np.linspace(0,2*pi,100)])
    poly = Poly3DCollection([circle], alpha = 0.3, facecolor='gray', linewidths=1)
    ax.add_collection3d(poly)
    
    ax.plot([xan,xan2],[yan,yan2],[zan,zan2],'-',color='gray',alpha=0.5)
    
    xp, yp, zp = perigee = vec(p/(1+e),0,0)@axes
    xa, ya, za = apogee  = vec(-p/(1-e),0,0)@axes
    ax.plot([xa,xp],[ya,yp],[za,zp],'-',color='#1F77B4',alpha=0.3)
    ax.plot([xp],[yp],[zp],'.',color='#1F77B4')
    
    ua = radcir/2
    ax.plot([0,ua],[0,0],[0,0],'-',color='black',lw=1)
    ax.plot([0,0],[0,ua],[0,0],'-',color='black',lw=1)
    ax.plot([0,0],[0,0],[0,ua],'-',color='black',lw=1)
    
    ax.plot([0],[0],[0],'o',color='orange')
    
################################################################

