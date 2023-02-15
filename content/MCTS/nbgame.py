# tools for interactive play in the notebook


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import asyncio

from mtcs import UTCSearch, MCPlay


async def waitCommand():
    global command
    command = None
    while command is None:
        await asyncio.sleep(0.01)
    return command


class board():
    def __init__(self,game):
        fig = plt.figure(figsize=(6,6))
        self.fig = fig

        self.G = game
        
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.capture_scroll = False

        def on_press(event):
            global command
            command = event.key
            #self.info(command)
        
        def onclick(event):
            global command, move
            command = 'move'
            #self.info(command)
            if self.G.name == 'connect':
                move = int(round(event.xdata))
            if self.G.name == 'othello':
                move = int(round(event.xdata)) + int(round(event.ydata))*self.G.n
        
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
        self.fig.canvas.draw()

    def info(self, msg):
        self.fig.suptitle(msg)
        self.fig.canvas.draw()

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
        #ax2.set_title(f'{str(sum(n))}')
        ax3.clear()
        ax3.bar(np.arange(len(a)),n,color='red')
        ax2.set_axis_off()
        ax3.set_axis_off()
        self.fig.canvas.draw()
    
        

def replay(pos,s):
    g = pos.reset()
    for a in pos.moves[:s]:
        g = g.action(a)
    return g

def retract(pos):
    g = replay(pos, max(0,len(pos.moves)-2))
    #g.moves = pos.moves
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

async def Human(g, board):
    global move
    if len(g.valid) == 1:
        print('forced move')
        board.info('your turn (forced move)')
        time.sleep(3)
        g = g.action(g.valid[0])
        g.eval.append((-g.turn,-1))
        board.info('')
        return g
    board.info('your turn')
    while True:
        comm = await waitCommand()

        if comm == 'move':
            g = g.action(move)
            g.eval.append((-g.turn,-1))
            board.info('')
            board.fig.canvas.draw()
            return g

        if comm == 'b':
            g = retract(g)
            g.draw(ax)
            board.drawvals(g)
            board.fig.canvas.draw()

        if comm == 'left':
            g = back(g)
            g.draw(ax)
            board.drawvals(g)
            board.fig.canvas.draw()

        if comm == 'right':
            g = forward(g)
            g.draw(ax)
            board.drawvals(g)
            board.fig.canvas.draw()


        if comm == 'x':
            g.resigned = True
            g.draw(ax)
            board.fig.canvas.draw()
            return g



def metaplay(board, g, dW, dB):
    engine = {1: dW['name'], -1: dB['name'], 0: 'nobody'}
    color = {1:'white', -1: 'black', 0: 'nobody'}

    async def play():
        nonlocal g
        current_player = dW['fun']
        next_player    = dB['fun']
        curr_async     = dW['isasync']
        next_async     = dB['isasync']
        while True:
            if curr_async:
                g = await current_player(g)
            else:
                g = current_player(g)

            #board.info(str(g.moves))
            current_player, next_player = next_player, current_player
            curr_async, next_async = next_async, curr_async
            g.draw(board.ax)
            board.drawvals(g)
            board.fig.canvas.draw()

            if g.terminal:
                board.info(f'{color[g.winner]} ({engine[g.winner]}) wins!')
                break
            if g.resigned:
                board.info(f'{color[g.turn]} ({engine[g.turn]}) resigned')
                break

    loop = asyncio.get_event_loop()
    loop.create_task(play())