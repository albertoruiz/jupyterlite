# tools for interactive play in the notebook


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

from mtcs import UTCSearch, MCPlay

import time


class board():
    def __init__(self,game):
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        self.fig = fig

        self.G = game
        
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.capture_scroll = False

        self.command = None
        self.move    = None

        def on_press(event):
            self.command = event.key
            # print(f'command {self.command}')
        
        def onclick(event):
            self.command = 'move'
            if self.G.name == 'connect':
                self.move = int(round(event.xdata))
            if self.G.name == 'othello':
                self.move = int(round(event.xdata)) + int(round(event.ydata))*self.G.n
            # print(f'move {self.move}')
        
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.mpl_connect('button_press_event', onclick)

        gs = gridspec.GridSpec(16, 4)
        self.ax  = plt.subplot(gs[4:16,0:4])
        self.ax2 = ax2 = plt.subplot(gs[1:4,0:1])
        self.ax3 = ax3 = plt.subplot(gs[1:4,1:2])
        self.ax4 = ax4 = plt.subplot(gs[1:4,2:4])
        ax4.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        plt.tight_layout(pad=0,h_pad=0,w_pad=0)
        self.info('Monte Carlo Tree Search')
        game.draw(self.ax)
        plt.show()
        self.refresh()

    def info(self, msg):
        self.fig.suptitle(msg)
        self.refresh()

    def drawvals(self,g):
        ax4 = self.ax4
        ax4.clear()
        ax4.set_ylim(-0.1,1.1)
        ax4.set_axis_off()
        l1 = [v for t,v in g.eval[:g.current] if t == 1]
        l2 = [v for t,v in g.eval[:g.current] if t == -1]
        l = max(len(l1),len(l2),2)
        rx = [0,l-1]
        ax4.plot(rx,[0,0], color='black',lw=0.5,ls='dotted')
        ax4.plot(rx,[0.5,0.5], color='black',lw=0.5,ls='dotted')
        ax4.plot(rx,[1,1], color='black',lw=0.5,ls='dotted')
        ax4.plot(l1, '.-', markersize=10, color='gray')
        ax4.plot(l2, '.-', markersize=10, color='black')

    def drawprobs(self,probs):
        ax2 = self.ax2
        ax3 = self.ax3
        ax2.clear()
        a,n,w = zip(*probs)
        an = len(a)
        ax2.bar(np.arange(an),np.array(w)/n)
        ax2.plot([-1,an],[0.5,0.5], color='black',lw=0.5)
        ax2.plot([-1,an],[0,0], color='black',lw=0.5)
        ax2.plot([-1,an],[1,1], color='black',lw=0.5)
        ax2.set_ylim(-0.1,1.1)
        ax3.clear()
        ax3.bar(np.arange(len(a)),n,color='red')
        ax2.set_axis_off()
        ax3.set_axis_off()
        self.refresh()
    
    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.1)

    def waitCommand(self):
        self.command = None
        while self.command is None:
            self.fig.canvas.start_event_loop(0.1)
        # print(self.command)


def replay(pos,s):
    g = pos.reset()
    for a in pos.moves[:s]:
        g = g.action(a)
    return g

def retract(pos):
    g = replay(pos, max(0,len(pos.moves)-2))
    g.eval = pos.eval[:g.current-2]
    return g

def back(pos):
    g = replay(pos, max(0, pos.current-1))
    g.moves = pos.moves
    g.eval = pos.eval
    return g

def forward(pos):
    g = replay(pos, pos.current+1)
    g.moves = pos.moves
    g.eval = pos.eval
    return g

def Human(g, board):
    global move
    if len(g.valid) == 1:
        print('forced move')
        board.info('your turn (forced move)')
        time.sleep(3)
        g = g.action(g.valid[0])
        g.eval.append((-g.turn,-1))
        board.info('')
        return g
    if g.current == len(g.moves):
       board.info('your turn')
    else:
        board.info('review game')
    while True:
        board.waitCommand()
        
        if board.command == 'move':
            g = g.action(board.move)
            g.eval.append((-g.turn,-1))
            board.info('')
            board.refresh()
            return g

        if board.command == 'b':
            g = retract(g)
            g.draw(board.ax)
            board.drawvals(g)
            board.refresh()

        if board.command == 'left':
            g = back(g)
            g.draw(board.ax)
            board.drawvals(g)
            board.refresh()

        if board.command == 'right':
            g = forward(g)
            g.draw(board.ax)
            board.drawvals(g)
            board.refresh()

        if board.command == 'escape':
            g.resigned = True
            g.draw(board.ax)
            board.refresh()
            return g



def play(board, g, dW, dB):
    engine = {1: dW['name'], -1: dB['name'], 0: 'nobody'}
    color = {1:'white', -1: 'black', 0: 'nobody'}

    def metaplay():
        nonlocal g
        current_player = dW['fun']
        next_player    = dB['fun']
        curr_async     = dW['isasync']
        next_async     = dB['isasync']
        while True:
            g = current_player(g)

            current_player, next_player = next_player, current_player
            curr_async, next_async = next_async, curr_async
            g.draw(board.ax)
            board.drawvals(g)
            board.refresh()

            if g.terminal:
                board.info(f'{color[g.winner]} ({engine[g.winner]}) wins!')
                break
            if g.resigned:
                board.info(f'{color[g.turn]} ({engine[g.turn]}) resigned')
                break

        print(g.moves)

        board.waitCommand()
        
        if board.command == 'up':
            greview = g.reset()
            greview.eval = g.eval
            greview.moves = g.moves
            greview.draw(board.ax)
            board.drawvals(greview)
            board.refresh()
            Human(greview,board)

    metaplay()

