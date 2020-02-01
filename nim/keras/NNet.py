import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
from keras.callbacks import TensorBoard
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .nimNNet import nimNNet as onnet
import torch

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        #log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if sys.platform == 'darwin':
            log_dir = '''/Users/mettinger/Google Drive/tensorboardLogs/'''
        else:
            log_dir = '''/content/drive/My Drive/tensorboardLogs/'''
            
        tensorboard = TensorBoard(log_dir=log_dir)

        self.nnet.model.fit(x = input_boards, 
                            y = [target_pis, target_vs], 
                            batch_size = args.batch_size, 
                            epochs = args.epochs,  
                            callbacks=[tensorboard])

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        #start = time.time()

        # preparing input
        board = board[np.newaxis, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Saving: " + filepath)
    
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
