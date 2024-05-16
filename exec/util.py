
import torch

class ModelSaver:
    def __init__(self, path) -> None:
        self.lastMax = -1.0
        self.path = path

    def save(self, model, acc, name):
        if acc > self.lastMax:
            self.lastMax = acc
            torch.save(model.state_dict(), self.path+name)