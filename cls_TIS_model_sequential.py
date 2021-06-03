import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Based on TISRover Experiment 1
        # 1 input image channel, 70 output channels
        self.features = nn.Sequential(nn.Conv2d(1, 70, (7, 4)),
                                      nn.ReLU(),
                                      nn.MaxPool2d((3, 1)),
                                      nn.Dropout2d(p=0.2),
                                      nn.Conv2d(70, 100, (3, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d((3, 1)),
                                      nn.Dropout2d(p=0.2),
                                      nn.Conv2d(100, 150, (3, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d((3, 1)),
                                      nn.Dropout2d(p=0.2))
        self.classifier = nn.Sequential(nn.Linear(900, 512),
                                        nn.Dropout2d(p=0.2),
                                        nn.ReLU(),
                                        nn.Linear(512, 2))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':

    model = Net()
    x = torch.randn((1, 1, 203, 4))
    out = model(x)
    #print(out.shape)
    #print(out)
    #params = list(model.parameters())
    #print(len(params))
    #print(params[7].size())  # conv1's .weight
    #print(model.size())
