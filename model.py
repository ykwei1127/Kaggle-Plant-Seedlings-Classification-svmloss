import torch.nn as nn
import math

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()

        # input_shape  = (3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # ouput_shape = (64,224,224)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # ouput_shape = (64,112,112)

        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # ouput_shape = (128,112,112)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # ouput_shape = (128,56,56)

        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # ouput_shape = (256,56,56)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # ouput_shape = (256,56,56)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # ouput_shape = (256,28,28)

        self.cnn5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # ouput_shape = (512,28,28)
        self.relu5 = nn.ReLU()
        self.cnn6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # ouput_shape = (512,28,28)
        self.relu6 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # ouput_shape = (512,14,14)

        self.cnn7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # ouput_shape = (512,14,14)
        self.relu7 = nn.ReLU()
        self.cnn8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # ouput_shape = (512,14,14)
        self.relu8 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2) # ouput_shape = (512,7,7)

        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.relu9 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu10 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(in_features=4096, out_features=12)
        # self.ouput = nn.Softmax(dim=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()



    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.maxpool3(out)

        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.cnn6(out)
        out = self.relu6(out)
        out = self.maxpool4(out)

        out = self.cnn7(out)
        out = self.relu7(out)
        out = self.cnn8(out)
        out = self.relu8(out)
        out = self.maxpool5(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu9(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu10(out)
        out = self.drop2(out)
        out = self.fc3(out)
        # out = self.ouput(out)

        return out