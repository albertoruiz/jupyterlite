from sympy import *
init_printing(pretty_print=True, use_latex='mathjax')

from sympy.physics.mechanics import dynamicsymbols, init_vprinting, vlatex
init_vprinting(pretty_print=True)

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from IPython.display import display

from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')

def show(expr, name=None):
    if name:
        display(Eq(Symbol(name),expr))
    else:
        display(expr)

def disp(expr,name=None):
    if type(expr) == list:
        expr = Matrix(expr)
    return Eq(Symbol(name), UnevaluatedExpr(expr))

def disp3(a,b,c):
    return Eq(a, Eq(b, c))

def data3d(line, x,y,z):
    line.set_data(x,y)
    line.set_3d_properties(z)


t = Symbol('t')


def ldisplay(x, name=None):
    if name is None:
        pre = ""
    else:
        pre = name + "="
    print(f'$${pre}{vlatex(x)}$$')

def lldisplay(xs):
    def translate(ec):
        return vlatex(ec.lhs) + '&=' + vlatex(ec.rhs)
    s = '$$\\begin{{aligned}}\n{}\n\\end{{aligned}}\n$$'
    ecs = [translate(x) for x in xs]
    r = '\\\\'.join(ecs)
    print(s.format(r))

def dynsyms(names):    
    return [ Function(s)(t) for s in names ]

def vector(n, name, equal=False):
    if n == 1:
        return Matrix([symbols(name)])
    if equal:
        return Matrix([symbols(name) for _ in range(n)])
    return Matrix([symbols(f'{name}{k}') for k in range(1,n+1)])  

def mul(a,b): return a.multiply_elementwise(b)

def posvel(n, d):
    P = Matrix([ dynsyms(f'x{n} y{n} z{n}'.split(' ')[:d])
                 for n in range(1,n+1) ])
    V = diff(P,t)
    return P,V

def mkGen(pos, vs):
    return {v:w for v,w in zip(Matrix(pos), vs) }

def val(u,v):
    return dict(zip(u,v))

def totup(x): return tuple([ tuple(z) for z in x])

def bracket(f,g,can):
    (Q,P) = can
    return sum([diff(f,q)*diff(g,p) - diff(f,p)*diff(g,q) for q,p in zip(Q,P)])

class Dynamics:
    def __init__(self, T, V, Gen, Q, sus, cons=[], F=sympify(0), onlyEcs=False, alsoH=False):
    
        self.Q = Q
    
        self.GenD = GenD = {diff(z,t) : diff(v,t) for z,v in Gen.items()}

        self.V = pot = V.subs(GenD).subs(Gen)

        self.T = kin = T.subs(GenD).subs(Gen)

        self.L = L   = (kin - pot).subs(sus).expand()

        self.D =   D = [diff(q,t) for q in Q]
        self.D2 = D2 = [diff(q,t,t) for q in Q]

        self.Mul = Mul = dynsyms([str(x) for x in symbols(f'lamda_1:{1+len(cons)}')])

        self.const = [c.subs(Gen).subs(sus) for c in cons]

        self.F = F.subs(GenD).subs(Gen).subs(sus)
        
        ecs = [Eq(diff(diff(L,dq),t), diff(L,q) - sum([m*diff(c,q) for m,c in zip(Mul,self.const)]) - diff(self.F,dq), evaluate=False)
                  for q,dq in zip(Q,D)]
        ecs = ecs + [Eq(diff(c,t,2).expand(),0) for c in self.const]
        
        self.ecsL = ecs
        #self.ecsL = [Eq(v,0) if e==True else e for e,v in zip (ecs,D2)]  # veremos...

        

        self.Vel = Vel = dynsyms([str(x) for x in symbols(f'u_1:{1+len(Q)}')])
        self.susv = susv = { diff(q,t) : v for q,v in zip(Q,Vel)} 
        
                
        self.m = m = [[(e.lhs - e.rhs).coeff(q).subs(susv).simplify() for q in D2+Mul] for e in self.ecsL]

        self.f = f = [(e.rhs - e.lhs).subs({q:0 for q in D2+Mul}).subs(susv).simplify() for e in self.ecsL]

        if onlyEcs:
            return

        mf = lambdify(Q+Vel+[t],(totup(m),tuple(f)),'math')

        def dotL(args,t):
            m,f = mf(*args,t)
            return np.hstack([args[-len(args)//2:] , la.solve(m,f)[:len(args)//2]])

        self.dotL = dotL

        self.nconst = lambdify(Q,self.const,'math')

        self.coords = lambdify([t]+Q, [Gen[x].subs(sus) for x in Gen.keys()],'math')
        
        if not alsoH:
            return
        
        self.P = P = dynsyms(symbols(f'p_1:{1+len(Q)}'))
        self.Pec = Pec = [Eq(p,diff(L,dq).simplify()) for p,dq in zip(P,D)]
        self.H0 = H0 = sum([p.rhs*d for p,d in zip(Pec,D)]) - L
        self.cansus = solve(Pec,D)
        self.H = H = H0.subs(self.cansus).simplify()
        self.ecsH = [Eq(diff(z,t),bracket(z,H,(Q,P))) for z in Q+P]

        num = lambdify(Q+P+[t],[e.rhs for e in self.ecsH],'math')
        def dotH(args,t):
            return num(*args,t)
        self.dotH = dotH



def nsolve(dot, T, dt, q0):
    t = np.linspace(0,T,round(T/dt))
    r = odeint(dot,q0,t)
    return np.hstack([ t.reshape(-1,1), r])

def mkAnim(sol,sys,prepareFunc,fps,frames):
    fig, drawFunc = prepareFunc()
    def animate(i):
        drawFunc(i, sol[i,0], *sys.coords(*sol[i,:1+len(sys.Q)]))
        return ()
    return animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps, blit=True)

def graph(solution,system, moments=False):
    plt.figure(figsize=(6,3))
    t,*xs = solution.T

    n = len(xs)//2
    for x,v in zip(xs,system.Q):
        plt.plot(t,x,label=f'${vlatex(v)}$')
    for x,v in zip(xs[n:], system.P if moments else system.D):
        plt.plot(t,x,label=f'${vlatex(v)}$')
    plt.grid()
    plt.legend()
    plt.xlabel('t');


import pickle

def save(filename, thing):
    f = open(f'{filename}.pkl','wb')
    pickle.dump(thing,f)
    f.close()

def load(filename):
    f = open(f'{filename}.pkl','rb')
    thing = pickle.load(f)
    f.close()
    return thing


# OJO, con una fila se ve columna (?!)
def rot3(ang):
    c = cos(ang)
    s = sin(ang)
    return Matrix([[c, -s, 0]
                  ,[s,  c, 0]
                  ,[0,  0, 1]])

def rot1(ang):
    c = cos(ang)
    s = sin(ang)
    return Matrix([[1, 0,  0]
                  ,[0, c, -s]
                  ,[0, s,  c]])

def rot2(ang):
    c = cos(ang)
    s = sin(ang)
    return Matrix([[ c, 0, s]
                  ,[ 0, 1, 0]
                  ,[-s, 0, c]])

# y que esto funciona pero hay que revisarlo
def EulerAnglesToRotation(ω, Ω, i):
    return (rot3(Ω) @ rot1(i) @ rot3(ω)).T


# para recuperar las coordenadas cartesianas después de integrar
# las generalizadas, y opcionalmente añadir puntos de referencia constantes
# útiles en las animaciones
def expandCoords(sol, sys, constants):
    c = np.array(constants).flatten()
    return np.array([[s[0], *sys.coords(*s[:1+len(sys.Q)]), *c] for s in sol])

