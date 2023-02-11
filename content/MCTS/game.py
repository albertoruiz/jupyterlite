#!/usr/bin/env python

# Experimentos con MTCS

from connect4 import connect
from othello import othello
from mtcs import UTCSearch, MCPlay

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

from matplotlib import gridspec
import time

import cProfile


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




def drawprobs(probs):
    ax2.clear()
    a,n,w = zip(*probs)
    an = len(a)
    ax2.bar(np.arange(an),np.array(w)/n)
    ax2.plot([-1,an],[0.5,0.5], color='black',lw=0.5)
    ax2.plot([-1,an],[0,0], color='black',lw=0.5)
    ax2.plot([-1,an],[1,1], color='black',lw=0.5)
    ax2.set_ylim(-0.1,1.1)
    #ax2.set_title(f'{str(sum(n))}')
    ax2.set_axis_off()
    ax3.clear()
    ax3.bar(np.arange(len(a)),n,color='red')
    ax3.set_axis_off()
    refresh()

def drawvals(g):
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

def refresh():
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)




def waitCommand():
    global command
    command = None
    while command is None:
        fig.canvas.start_event_loop(0.1)
    return command

def on_press(event):
    global command
    command = event.key
    #print(command)

def onclick(event):
    global command, move
    command = 'move'
    if args.game == 'connect':
        move = int(round(event.xdata))
    if args.game == 'othello':
        move = int(round(event.xdata)) + int(round(event.ydata))*args.size


def Human(g):
    global move
    if len(g.valid) == 1:
        print('forced move')
        info('your turn (forced move)')
        time.sleep(3)
        g = g.action(g.valid[0])
        g.eval.append((-g.turn,-1))
        info('')
        return g
    info('your turn')
    while True:
        comm = waitCommand()

        if comm == 'move':
            g = g.action(move)
            g.eval.append((-g.turn,-1))
            info('')
            return g

        if comm == 'b':
            g = retract(g)
            g.draw(ax)
            drawvals(g)
            refresh()

        if comm == 'left':
            g = back(g)
            g.draw(ax)
            drawvals(g)
            refresh()

        if comm == 'right':
            g = forward(g)
            g.draw(ax)
            drawvals(g)
            refresh()


        if comm == 'a':
            g.resigned = True
            g.draw(ax)
            refresh()
            return g




import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--game', help="selected game (othello, connect)", type=str,default='connect')
parser.add_argument('--nowait', help="start next match immediately", action='store_true')
parser.add_argument('--nodes', type=int, default=10000, help='max node search')
parser.add_argument('--rollout', type=int, default=100, help='rollout trials in pure monte-carlo')
parser.add_argument('--timeout', type=int, default=10, help='max thinking time')
parser.add_argument('--white', type=str, default='mcts', help='first player')
parser.add_argument('--black', type=str, default='human', help='first player')
parser.add_argument('--rows',  type=int, default='6', help='rows of the board')
parser.add_argument('--cols',  type=int, default='7', help='columns of the board')
parser.add_argument('--size',  type=int, default='8', help='size of square board')
parser.add_argument('--connect',  type=int, default='4', help='number of pieces to connect')
args = parser.parse_args()

games = {'othello': (othello, args.size),
         'connect': (connect, args.cols, args.rows, args.connect)}

players = {'mcts':  lambda g: UTCSearch(g, args.nodes, args.timeout, drawprobs),
           'pure':  lambda g: MCPlay(g,args.rollout),
           'human': lambda g: Human(g)}


plt.ion()
fig = plt.figure(figsize=(6,6))
#fig.canvas.set_window_title('MCTS')
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('button_press_event', onclick)

def info(msg):
    ax.set_title(msg) #setVisible(False)

gs = gridspec.GridSpec(4, 4)
ax  = plt.subplot(gs[1:4,0:4])
ax2 = plt.subplot(gs[0,0:1])
ax3 = plt.subplot(gs[0,1:2])
ax4 = plt.subplot(gs[0,2:4])
ax4.set_axis_off()
plt.tight_layout(pad=0,h_pad=0,w_pad=0)

color = {1:'white', -1: 'black', 0: 'nobody'}


while True:
    g = games[args.game]
    g = g[0](*g[1:])
    print(g.eval)
    #g.moves = [26, 25, 16, 32, 30, 24, 18, 8, 31, 17, 9, 2, 33, 27, 10, 3, 5, 22, 34, 19, 12, 4, 1, -1, 28, 35, 29, 11, 7, 0, 23, -1, 6, -1, 13]
    #g = g.replay(14)

# impresionante
#[29, 21, 13, 37, 43, 5, 14, 19, 30, 34, 41, 7, 44, 31, 10, 26, 20, 48, 18, 2, 9, 17, 42, 0, 40, 32, 1, 52, 6, 11, 25, 8, 22, 16, 38, 15, 24, 33, 56, 39, 12, 4, 23, 49, 57, -1, 53, 62, 61, 60, 3, 50, 59, 58, 45, 51, 46, 47, 63, -1, 55, -1, 54]

# hay que analizar este, que da un vuelco al final...???
#[29, 19, 42, 43, 34, 45, 37, 46, 52, 59, 54, 63, 21, 50, 58, 57, 53, 60, 18, 26, 20, 44, 61, 62, 33, 24, 40, 11, 47, 55, 2, 39, 12, 38, 51, 4, 30, 31, 3, 10, 17, 1, 22, 23, 25, 5, 13, 8, 16, 9, 0, 32, 41, 14, 15, 6, 7, 48, 49, -1, 56]


# y este también!
# [29, 21, 13, 26, 34, 42, 18, 19, 50, 37, 20, 5, 30, 23, 22, 31, 46, 44, 39, 47, 15, 7, 6, 55, 43, 4, 38, 17, 12, 3, 9, 58, 11, 2, 14, 10, 45, 52, 49, 56, 53, 8, 0, 1, 54, 63, 33, 32, 25, 24, 16, 62, 40, -1, 57, 51, 59, 60, 41, -1, 61, -1, 48]



    g.draw(ax)
    refresh()

    white = players[args.white]
    black = players[args.black]
    engine = {1: args.white, -1: args.black, 0: 'nobody'}

    current_player = white
    next_player    = black

    while True:
        g = current_player(g)
        current_player, next_player = next_player, current_player
        g.draw(ax)
        drawvals(g)
        refresh()
        
        if g.terminal:
            info(f'{color[g.winner]} ({engine[g.winner]}) wins!')
            break
        if g.resigned:
            info(f'{color[g.turn]} ({engine[g.turn]}) resigned')
            break


    print(g.moves)
    if not args.nowait:
        print(args)
        comm = waitCommand()
        if comm == 'escape':
            break
        if comm == 'up':
            greview = g.reset()
            greview.eval = g.eval
            greview.moves = g. moves
            greview.draw(ax)
            drawvals(greview)
            refresh()
            info('review game')
            Human(greview)

    info('')

    args.white, args.black = args.black, args.white
    engine = {1: args.white, -1: args.black, 0: 'nobody'}

