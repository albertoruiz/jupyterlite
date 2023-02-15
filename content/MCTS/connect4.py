
import numpy as np
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import time


def mkWinKers(w):
    kers  = [np.ones((w,1),int)]
    kers += [np.ones((1,w),int)]
    kers += [np.eye(w).astype(int)]
    kers += [np.flipud(np.eye(w).astype(int))]
    return kers


class connect:
    def __init__(self,n,h,w,init=True):
        self.name = 'connect'
        self.n = n
        self.h = h
        self.w = w
        self.resigned = False
        if init:
            self.board = np.zeros((h,n))
            self.turn = +1
            self.kers = mkWinKers(w)
            self.terminal = False
            self.resigned = False
            self.moves = []
            self.valid = list(range(n))
            self.eval = []
            self.current = 0


    def valid_actions(self):
        assert not self.terminal
        return self.valid


    def action(self,k):
        g = connect(self.n, self.h, self.w, init=False)
        g.board = self.board.copy()
        f = np.argmax(abs(g.board[:,k])) - 1
        g.board[f,k] = self.turn
        g.turn = -self.turn
        g.kers = self.kers
        g.moves = self.moves[:self.current] + [k]
        
        g.eval = self.eval[:self.current]
        g.current = self.current + 1
        g.valid = np.where(g.board[0]==0)[0]
        g.check_terminal()
        return g


    def check_terminal(self):
        conv = np.concatenate([ convolve2d(self.board, k).flatten() for k in self.kers ])
        self.terminal = False
        if conv.max() == self.w:
            self.winner = +1
            self.terminal = True
            return
        if conv.min() == -self.w:
            self.winner = -1
            self.terminal = True
            return
        if len(self.valid) == 0:
            self.winner = 0
            self.terminal = True


    def reset(self):
        return connect(self.n, self.h, self.w, init=True)



    def draw(self, ax):
        arr = self.board
        h,c = arr.shape
        ax.clear()
        #ax.set_facecolor('whitesmoke')
        ax.set_facecolor('lightgray')
        color = {+1:'white',-1:'gray',0:'gainsboro'}
        for j in range(h):
            for k in range(c):
                ax.add_patch(mpatches.Circle((k,j), radius=0.45,color=color[arr[j,k]], ec='gray'))
        if not self.terminal:
            ax.add_patch(mpatches.Circle((-1.5,0), radius=0.3,color=color[self.turn], ec='gray'))
        ax.set_xlim(-0.5,c+0.5)
        ax.set_ylim(h+0.5,-0.5)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

