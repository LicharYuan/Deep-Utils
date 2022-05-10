from abc import ABCMeta, abstractmethod

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args
        self.start_epoch = 0
        self.end_epoch = 100
    
    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def before_train(self):
        pass

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.end_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    @abstractmethod
    def train_in_iter(self):
        pass

    def after_epoch(self):
        pass



    

    


        