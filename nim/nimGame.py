
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import copy
import math

# SUM OF INTEGERS FROM 1 TO N
def sumIntegers(n):
        return int((n * ((n + 1)/2.)))

# FROM AN INTEGER ACTION CODE, CALCULATE THE PILE SIZE AND REMOVAL NUMBER
# CODING: FOR A SIZE=1 PILE THERE IS EXACTLY ONE POSSIBLE MOVE.
#         FOR A SIZE=2 PILE THERE ARE EXACTLY TWO POSSIBLE MOVES, ETC.
def actionDecode(action):
    action = action + 1
    quadraticSolution = (-1 + math.sqrt(1 + (8 * action)))/2.
    pileSize = math.ceil(quadraticSolution)
    removeSize = int(action - ((pileSize) * ((pileSize-1)/2.)))

    return pileSize, removeSize

# CALCULATE THE NIMBER ASSOCIATED WITH A STATE
def stateToNimber(pileList, shortNotation=True):
    if shortNotation:
        pileList = [i % 2 for i in pileList]
        longList = []
        for index, count in enumerate(pileList):
            longList.extend([index + 1 for i in range(count)])
        return stateToNimber(longList, False)
    else:  
        if len(pileList) == 0:
            return 0
        elif len(pileList) == 1:
            return pileList[0]
        elif len(pileList) == 2:
            a = format(pileList[0], 'b')
            b = format(pileList[1], 'b')
            
            if len(a) > len(b):
                b = b.zfill(len(a))
            else:
                a = a.zfill(len(b))
                
            c = ''
            for i in range(len(a)):
                c = c + str((int(b[i]) + int(a[i])) % 2)
            return int(c,2)
                
        else:
            return stateToNimber([pileList[0], stateToNimber(pileList[1:], False)], False)

class nimGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, config):
        self.maxPileSize = config['maxPileSize']
        self.maxNumPile = config['maxNumPile']
        self.initialState = config['initialState']
        self.randomInitial = config['randomInitial']
        self.numAction = sumIntegers(self.maxPileSize)
        self.state = copy.deepcopy(self.initialState)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        if self.initialState is not None:
            return self.initialState
        else:
            return np.random.randint(low=0, high=self.maxNumPile, size=(self.maxPileSize,))

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (1, self.maxPileSize)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.numAction

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pileSize, removeSize = actionDecode(action)
        nextBoard = copy.copy(board)
        nextBoard[pileSize - 1] = nextBoard[pileSize - 1] - 1
        if removeSize < pileSize:
            newPileSize = pileSize - removeSize
            nextBoard[newPileSize - 1] = nextBoard[newPileSize - 1] + 1
        
        return nextBoard, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """

        
        validMoves = np.zeros(self.getActionSize())
        for action in range(self.getActionSize()):
            pileSize, _ = actionDecode(action)
            if board[pileSize - 1] > 0:
                validMoves[action] = 1
        

        #validMoves = np.ones(self.getActionSize())
        return validMoves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if sum(board) > 0:
            return 0
        else:
            #print("Winner: " + str(-player))
            #return -1
            return -player

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board,pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(list(board))


    @staticmethod
    def display(board):
        print("Board: " + str(board))
        print("Nimber: " + str(stateToNimber(board)))
