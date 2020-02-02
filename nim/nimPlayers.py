class HumanNimPlayer():
    def __init__(self, game):
        self.game = game

    def sumIntegers(self, n):
        return int((n * ((n + 1)/2.)))

    def pileAction2Integer(self, pileSize, removeSize):
        pileSum = self.sumIntegers(pileSize - 1)
        integerAction = pileSum + removeSize
        return int(integerAction - 1)

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        
        while True: 
            # Python 3.x
            a = input("Enter your move: ")
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(',')]
            a = self.pileAction2Integer(x,y)
            if valid[a]:
                break
            else:
                print('Invalid')

        return a