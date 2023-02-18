from connect4 import connect
from othello import othello
from mtcs import UTCSearch, MCPlay

import metagame

pure  = lambda n: dict(name='pure',  isasync=False, fun=lambda g: MCPlay(g, n))
mcts  = lambda n, time, board: dict(name='mcts',  isasync=False, fun=lambda g: UTCSearch(g, n, time, board.drawprobs))
human = lambda board: dict(name='human', isasync=True,  fun=lambda g: metagame.Human(g, board))

game = connect(7,6,4)
#game = othello(6)
B = metagame.board(game)


#metagame.play(board=B, g=game, dW=human(B), dB=human(B))
#metagame.play(board=B, g=game, dW=pure(100), dB=mcts(10000,5,B))
metagame.play(board=B, g=game, dW=human(B), dB=mcts(10000,5,B))

