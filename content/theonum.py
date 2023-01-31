# Simple Number Theory functions
# Alberto Ruiz 2019-23

# 03c0   π
# 025b   ɛ
# 20d7    ⃗

"α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω" 
"Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ   Σ Τ Υ Φ Χ Ψ Ω"

import math

#-------------------------------------------------------

def isqrt(n, trace=False, t0=1):
    assert n>=0, 'isqrt'
    if n==0: return (0,0)
    s = len(str(n))
    if trace: print(f'{s:3} digits')
    x = t0*10**(s//2)
    D = n-x*x
    dxa = 0
    for k in range(40):
        dx = D//(2*x)
        if dx==0 or dxa==-dx: break
        r  = D-2*x*dx
        D = r-dx*dx
        x += dx
        if trace: print(f'{len(str(abs(D))):3} {x}')
        dxa = dx
    if trace: print(f'{k} Newton steps')
    return x,D

#--------------------------------------------------------

def 𝜏(n):
    return list(range(1,n+1))


def fact(n):
    return math.prod(𝜏(n))


def div(k,n):
    return n%k==0

def D(𝑛):
    return [k for k in 𝜏(n) if div(k,n)]


def cop(a,b):
    return math.gcd(a,b)==1

def egcd(a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    gcd = b
    return gcd, x, y

def inv(a,n):
    assert cop(a,n)
    _,x,_ = egcd(a,n)
    return x % n


# ojo, deben ser pairwise coprime
def crt(rn):
    if len(rn)==1:
        return rn[0][0] % rn[0][1], rn[0][1]
    (r1,n1),(r2,n2),*rns = rn
    d,x,y = egcd(n1,n2)
    return crt([(r1*y*n2 + r2*x*n1, n1*n2)] + rns)


def T(n):
    return [k for k in 𝜏(n) if cop(k,n)]

def 𝜑(n):
    from sympy.ntheory import factorint
    ps = factorint(n).keys()
    r = n
    for p in ps:
        r =  r * (p-1) // p
    return r


#----------------------------------------------------------------------

def cf_expansion(n, d):
    e = []

    q = n // d
    r = n % d
    e.append(q)

    while r != 0:
        n, d = d, r
        q = n // d
        r = n % d
        e.append(q)

    return e


def convergents(e):
    n = [] # Nominators
    d = [] # Denominators

    for i in range(len(e)):
        if i == 0:
            ni = e[i]
            di = 1
        elif i == 1:
            ni = e[i]*e[i-1] + 1
            di = e[i]
        else: # i > 1
            ni = e[i]*n[i-1] + n[i-2]
            di = e[i]*d[i-1] + d[i-2]

        n.append(ni)
        d.append(di)
        yield (ni, di)


def shcf(n,d):
    from IPython.display import Latex

    def shcfr(cs):
        if len(cs) == 1: return f'\\frac{{{1}}}{{{cs[0]}}}'
        return f'\\frac{{{1}}}{{{cs[0]}+{shcfr(cs[1:])}}}'

    b0,*bs = cf_expansion(n,d)
    if b0 == 0:
        sb0 = ''
    else:
        sb0 = f'{b0}+'

    return Latex(f'$${sb0}{shcfr(bs)}$$')


#----------------------------------------------------------------------

RSA100 = 35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667


