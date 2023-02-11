
import numpy as np
import time

######################################################################
# Pure Monte-Carlo

def rollout(root,n):
    P = 0
    N = 0
    for _ in range(n):
        g = root
        while not g.terminal:
            g = g.action(np.random.choice(g.valid_actions()))
        if g.winner > 0:
            P += 1
        if g.winner < 0:
            N += 1
    return P,N,n


def MCPlay(g,n):
    acts = g.valid_actions()
    pn = 0 if g.turn == 1 else 1
    probs = [rollout(g.action(a),n)[pn] for a in acts ]
    #print(probs)
    selected = acts[np.argmax(np.array(probs))]
    #print(f'Computer plays {selected}')
    g = g.action(selected)
    g.eval.append((-g.turn, max(probs)/n))
    return g


######################################################################
# Monte-Carlo Tree Search

TT = [0]

class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children  = []
        if not self.state.terminal:
            self.remaining = [state.action(a) for a in state.valid_actions()]
        self.N = 0
        self.Q = 0

def TreePolicy(v):
    while not v.state.terminal:
        if v.remaining:
            return Expand(v)
        else:
            v = BestChild(v, Cp)
    return v

def Expand(v):
    n = len(v.remaining)
    k = np.random.randint(n)
    new = Node(v.remaining[k], v)
    del v.remaining[k]
    v.children.append(new)
    return new

def BestChild(v,c):
    UTC = [ h.Q/h.N + c * np.sqrt(2 * np.log(v.N)/h.N) for h in v.children ]
    return v.children[np.argmax(UTC)]

def DefaultPolicy(s):
    while not s.terminal:
        s = s.action(np.random.choice(s.valid_actions()))
        TT[0] += 1
    #print(s.winner)
    return s.winner

def BackupNegamax(v, Delta):
    while v is not None:
        v.N += 1
        v.Q += Delta
        Delta = 1-Delta
        v = v.parent

def UTCSearch(s0, N, timeout, monitor=None):
    global fin
    TT[0] = 0
    if len(s0.valid) == 1:
        print('forced move')
        return s0.action(s0.valid[0])

    root = Node(s0,None)
    fin = False
    t0 = time.time()
    for k in range(N):
        v = TreePolicy(root)
        D = DefaultPolicy(v.state)

        if D == v.state.turn:
            D = 0
        elif D==-v.state.turn:
            D = 1
        else:
            D == 0.5

        BackupNegamax(v,D)

        if k%100==0 and monitor is not None:
            probs = sorted([(x.state.moves[-1], x.N, x.Q) for x in root.children])
            monitor(probs)

        if fin or (time.time()-t0)>timeout: break
    
    print(TT, k, time.time()-t0)
    probs = sorted([(x.state.moves[-1], x.N, x.Q) for x in root.children])
    monitor(probs)
    value = max([p[-1]/p[-2] for p in probs])
    g = BestChild(root,0).state
    g.eval.append((-g.turn,value))
    #print(g.eval)
    return g

Cp = 1/2**0.5

