import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        ## chanel output square-kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2*2 kernel size
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features
    
    def loss_function(self, output, gt):
        self.criterion = nn.MSELoss()

        loss = self.criterion(output, gt)
        return loss

if __name__ == "__main__":
    net = ConvNet()
    print("net: ", net)

    params = list(net.parameters())
    print("params: ", len(params), params[0].size())

    input = torch.randn(1, 1, 32, 32)
    # print("input = ", input.unsqueeze(0))
    out = net(input)
    print("out: ", out)

    net.zero_grad()
    # out.backward(torch.randn(1, 10))

    target = torch.randn(10)
    target = target.view(1, -1)
    loss = net.loss_function(out, target)
    print("loss = ", loss, loss.type, loss.item())

    # print("loss attribute = ", loss.grad_fn)
    # print("loss next function = ", loss.grad_fn.next_functions[0][0])

    # print("conv1 bias grad = ", net.conv1.bias.grad)
    # net.zero_grad()
    # print("conv1 bias grad = ", net.conv1.bias.grad)
    # loss.backward()
    # print("conv1 bias grad = ", net.conv1.bias.grad)

    # # update weight
    # learning_rate = 0.001
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)

    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    optimizer.zero_grad()
    out = net(input)
    loss = net.loss_function(out, target)
    loss.backward()
    optimizer.step()    