import Arena
from MCTS import MCTS

gameChoice = 2

if gameChoice == 0:
    from othello.OthelloGame import OthelloGame
    from othello.OthelloPlayers import *
    from othello.pytorch.NNet import NNetWrapper as NNet
elif gameChoice == 1:
    from tictactoe.TicTacToeGame import TicTacToeGame
    from tictactoe.TicTacToePlayers import *
    from tictactoe.keras.NNet import NNetWrapper as nn
elif gameChoice == 2:
    from nim.nimGame import nimGame
    from nim.nimPlayers import *
    from nim.keras.NNet import NNetWrapper as nn


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if gameChoice == 0:
    if mini_othello:
        g = OthelloGame(6)
    else:
        g = OthelloGame(8)
elif gameChoice == 1:
    g = TicTacToeGame()
elif gameChoice == 2:

    #initialState = np.array([1 for i in range(10)])
    initialState = None

    config = {'maxPileSize':10, 
              'maxNumPile':3, 
              'initialState': initialState}
    g = nimGame(config)

# all players
#rp = RandomPlayer(g).play

if gameChoice == 0:
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play
elif gameChoice == 1:
    hp = HumanTicTacToePlayer(g).play
elif gameChoice == 2:
    hp = HumanNimPlayer(g).play


# nnet players
if gameChoice == 0:
    n1 = NNet(g)
else:
    n1 = nn(g)

if gameChoice == 0:
    if mini_othello:
        n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
    else:
        n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
elif gameChoice == 1:
    n1.load_checkpoint('/Users/mettinger/github/alpha-zero-general/pretrained_models/tictactoe/keras','best-25eps-25sim-10epch.pth.tar')
elif gameChoice == 2:
    n1.load_checkpoint('''/Users/mettinger/Google Drive/models''', 'best.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    if gameChoice == 0:
        n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    elif gameChoice == 1:
        n1.load_checkpoint('/Users/mettinger/github/alpha-zero-general/pretrained_models/tictactoe/keras/','best-25eps-25sim-10epch.pth.tar')
    elif gameChoice == 2:
        n1.load_checkpoint('''/Users/mettinger/Google Drive/models''','best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

if gameChoice == 0:
    display = OthelloGame.display
    postAction = None
elif gameChoice == 1:
    display = TicTacToeGame.display
    postAction = None
elif gameChoice == 2:
    display = nimGame.display
    postAction = nimGame.postAction

arena = Arena.Arena(n1p, player2, g, display=display, postAction=postAction)

print(arena.playGames(2, verbose=True))
