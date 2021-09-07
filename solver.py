import torch

class Solver():
    def __init__(self,lr,lstm):
        self.lr = lr
        self.model = lstm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.losses = []
        self.eval_losses = []
