class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    '''
    def __getstate__(self):
        state = dict(self.__dict__)
        return state
    '''
