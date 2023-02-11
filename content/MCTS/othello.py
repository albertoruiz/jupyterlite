
# %%

import numpy as np
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import time

DIR = np.array([[ 1, 0],
                [ 1, 1],
                [ 0, 1],
                [-1, 1],
                [-1, 0],
                [-1,-1],
                [ 0,-1],
                [ 1,-1]])



class othello:
    
    def count(self, a, play=False):
        r,c = p = self.act2pos[a]
        B = self.board
        n = self.n
        yo = self.turn
        if play:
            B[r,c] = yo
        counter = 0
        for d in DIR:
            ok = True
            for l in range(1,self.n+1):
                y,x = p + d*l
                if not ((0<=x<n) and (0<=y<n)) or B[y,x] == 0:
                    ok = False
                    break
                if B[y,x]  == yo:
                    break
            if not ok:
                continue
            for k in range(1,l):
                y,x = p + d*k
                if play:
                    B[y,x] = yo
                counter += 1
        return counter
                
    
    
    def __init__(self,n,init=True):
        self.n = n
        self.resigned = False
        self.terminal = False
        self.prevpass = False
        if init:
            self.board = np.zeros((n,n),int)
            self.turn = +1
            self.terminal = False
            self.resigned = False
            self.moves = []
            self.board[n//2,n//2]= +1
            self.board[n//2-1,n//2-1]= +1
            self.board[n//2,n//2-1]= -1
            self.board[n//2-1,n//2]= -1
            self.pos2act = np.array([r*n+c for r in range(n) for c in range(n)]).reshape(n,n)
            self.act2pos = np.array([[a//n, a%n] for a in range(n*n)])
            self.valid = self.pos2act[self.board == 0]
            self.valid = [x for x in self.valid if self.count(x) > 0 ]
            self.eval = []
            self.current = 0


    def valid_actions(self):
        assert not self.terminal
        return self.valid


    def action(self,k):
        g = othello(self.n, init=False)
        g.pos2act = self.pos2act
        g.act2pos = self.act2pos
        g.board = self.board.copy()
        g.moves = self.moves + [k]
        g.eval = self.eval
        g.current = self.current + 1
        
        if k >=0:
            r,c = g.act2pos[k]
            g.turn = self.turn
            g.count(k,play=True)
            g.prevpass = False
            g.turn = -self.turn
        else: # pass
            g.turn = -self.turn
            g.prevpass = True
            if self.prevpass:
                g.terminal = True
                g.winner = np.sign(g.board.sum())
                return g
        
        empty = g.pos2act[g.board == 0]
        
        if len(empty)==0:
            g.terminal = True
            g.winner = np.sign(g.board.sum())
            return g
        
        g.valid = [x for x in empty if g.count(x) > 0]

        if not g.valid:
            g.valid = [-1]
        
        return g

    def reset(self):
        return othello(self.n, init=True)


    def draw(self, ax):
        arr = self.board
        h,c = arr.shape
        ax.clear()
        #ax.set_facecolor('whitesmoke')
        ax.set_facecolor('gainsboro')
        color = {+1:'white',-1:'gray',0:'gainsboro'}
        ec = {+1:'gray',-1:'dimgray'}
        for j in range(h):
            for k in range(c):
                if arr[j,k] != 0:
                    ax.add_patch(mpatches.Circle((k,j), radius=0.4,color=color[arr[j,k]], ec=ec[arr[j,k]]))
        if not self.terminal:
            for a in self.valid:
                y,x = self.act2pos[a]
                ax.add_patch(mpatches.Circle((x,y), radius=0.1,color='lightgray', ec='gray'))
            ax.add_patch(mpatches.Circle((-1.5,0), radius=0.3,color=color[self.turn], ec='gray'))
        ax.set_xlim(-0.5,c+0.5)
        ax.set_ylim(h+0.5,-0.5)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.hlines(np.arange(0,c+1)-0.5,-0.5,c-0.5,color='gray',lw=1)
        ax.vlines(np.arange(0,c+1)-0.5,-0.5,c-0.5,color='gray',lw=1)
        ax.text(-1.5,h-1,f'{self.board.sum()}')


g = othello(8)
