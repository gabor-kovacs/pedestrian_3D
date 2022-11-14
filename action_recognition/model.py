import torch
import torch.nn as nn
import torch.nn.functional as F

n_in = 36           # number of joints in openpose model
n_classes = 10      # number of actions

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_in, 500),
            nn.Dropout(p=0.2),
            nn.Hardswish(),
            nn.Linear(500, 1000),
            nn.Dropout(p=0.3),
            nn.Hardswish(),
            nn.Linear(1000, 200),
            nn.Dropout(p=0.2),
            nn.Hardswish(),
            nn.Linear(200, n_classes),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
  model = Net()
  model.eval()
  input = torch.rand(1,36)
  output = model(input)
  print(output)



