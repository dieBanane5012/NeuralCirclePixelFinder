import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()

        self.Convlayers = nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,64,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64,128,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )


        self.lin  = nn.Sequential(
            nn.Linear(128*8*8,256),
            nn.ReLU(),

            nn.Linear(256,256),
            nn.ReLU(),

            nn.Linear(256,2),
        )
    def forward(self,x):
        x = self.Convlayers(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
    

if __name__ == "__main__":
    mode = NN()
    print(mode)

    dummY_inp = torch.randn(1,1,64,64)

    output = mode(dummY_inp)
    print(f"output shape: {output.shape}")