
gameChoice = 2
serialFlag = False

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
import time
import copy

if sys.platform == 'darwin':
    #checkpoint = '''/Users/mettinger/Google Drive/models/'''
    checkpoint = '''/Users/mettinger/Desktop/old/'''
else:
    checkpoint = '''/content/drive/My Drive/models/'''
    
args = dotdict({
    'numIters': 100,              # Number of self-play and model fit rounds.
    'numEps': 300,               # Number of complete self-play games to simulate during a new iteration.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'tempThreshold': 15,        #
    'updateThreshold': 0.45,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': checkpoint,
    'load_model': True,
    'load_folder_file': (checkpoint,'checkpoint.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":

    print("Serial Flag: " + str(serialFlag))
    if serialFlag:
        if gameChoice == 0:
            g = Game(6)
        elif gameChoice == 1:
            g = TicTacToeGame()
        elif gameChoice == 2:

            #initialState = np.array([1 for i in range(10)])
            initialState = None  # None provides random board initial state

            config = {'maxPileSize':10, 
                    'maxNumPile':3, 
                    'initialState': initialState}

            g = nimGame(config)

        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
    
        c.learn()

    else:
        def selfPlayOnly(args):
            initialState = None
            config = {'maxPileSize':10, 
                    'maxNumPile':3, 
                    'initialState': initialState}
            g = nimGame(config)
            nnet = nn(g)
            coach_0 = Coach(g, nnet, args)
            for i in range(args.numIters):
                print("Self-play iteration: " + str(i))
                nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
                coach_0.selfPlay()

            
        process_selfPlay = multiprocessing.Process(target=selfPlayOnly, args=(args,))
        process_selfPlay.start()

        initialState = None
        config = {'maxPileSize':10, 
                'maxNumPile':3, 
                'initialState': initialState}
        g = nimGame(config)
        nnet = nn(g)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        coach_1 = Coach(g, nnet, args)
        for j in range(args.numIters):
            print("Train iteration: " + str(j))
            coach_1.loadTrainExamples()
            coach_1.trainNetwork()
            
