import torch.nn as nn

# 0.7003
class CovModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            Print("conv3"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*32*32, 256),
            nn.Dropout(0.5),
            Print("conv3"),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.conv3(x)
    
class CovModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            Print("1"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            Print("2"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            Print("3"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            Print("4"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            Print("5"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*32*32, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            Print("6"),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.conv3(x)
    
class CovModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3), # 96
            nn.BatchNorm2d(32),
            Print("1"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 48
            nn.BatchNorm2d(64),
            Print("2"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 24
            nn.BatchNorm2d(64),
            Print("3"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Print("4"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            Print("5"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*24*24, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            Print("6"),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.conv3(x)
    


class Print(nn.Module):
    def __init__(self, name):
        super(Print, self).__init__()
        self.name = name

    def forward(self, x):
        print(str(x.shape) + " " + self.name)
        return x