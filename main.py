
gameChoice = 2

if gameChoice == 0:
    from othello.OthelloGame import OthelloGame as Game
    from othello.pytorch.NNet import NNetWrapper as nn
elif gameChoice == 1:
    from tictactoe.TicTacToeGame import TicTacToeGame
    from tictactoe.keras.NNet import NNetWrapper as nn
elif gameChoice == 2:
    from nim.nimGame import nimGame
    from nim.keras.NNet import NNetWrapper as nn

from Coach import Coach
from utils import *
import numpy as np
import sys
import multiprocessing

if sys.platform == 'darwin':
    checkpoint = '''/Users/mettinger/Google Drive/models/temp/'''
else:
    checkpoint = '''/content/drive/My Drive/models/'''
    
args = dotdict({
    'numIters': 100,              # Number of self-play and model fit rounds.
    'numEps': 2,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.45,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': checkpoint,
    'load_model': True,
    'load_folder_file': (checkpoint,'checkpoint.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":

    if gameChoice == 0:
        g = Game(6)
    elif gameChoice == 1:
        g = TicTacToeGame()
    elif gameChoice == 2:

        initialState = np.array([1 for i in range(10)])

        config = {'maxPileSize':10, 
                  'maxNumPile':10, 
                  'initialState': initialState, 
                  'randomInitial':False}

        g = nimGame(config)

    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

    # multiprocessing stuff here
    pass
