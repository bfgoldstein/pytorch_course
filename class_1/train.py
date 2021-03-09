import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

from util import parser



class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output
    

def train(args, model, train_loader, criterion, optimizer, epoch):
    correct = 0
    total_count = 0
        
    model.train()
        
    for batch_idx, (input, target) in enumerate(train_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        # first zero the gradients from previous batch
        # we do not want to accumulate wrong gradients    
        optimizer.zero_grad()
        # do the inference with new batch
        output = model(input)
        # and calculate loss
        loss = criterion(output, target)
        # backpropagate the loss over the network to calculate new gradients
        loss.backward()
        # update the optimizer step
        optimizer.step()

        # we typically do not calculate the accuracy during training
        # the important metric is the loss (how it is drecreasing) and
        # the validation accuracy after each epoch
        # however, here is a way to calculate the accuracy during training
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_count += len(input)
        accuracy = correct / total_count
      
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                    
    return loss.item(), accuracy


def test(args, model, criterion, test_loader):
    loss_avg = 0
    correct = 0
    accs = []
    batch_count = 0
    
    model.eval()

    with torch.no_grad():
        for input, target in test_loader:
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                
            output = model(input)
            loss_avg += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_count += len(input)
            accs.append(correct / batch_count)

    loss_avg /= len(test_loader.dataset)
    acc_avg = correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss_avg, correct, len(test_loader.dataset), 100. * acc_avg))
    
    return loss_avg, acc_avg


# load the MNIST dataset into a dataloader object
# one for training and another one for testing
def loadData(args):    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=True, download=True,
                       transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=(args.gpu is not None))
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
                    batch_size=args.test_batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(args.gpu is not None))
    
    return train_loader, test_loader


def main():
    parser_obj = parser.getParser()
    args = parser_obj.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True 
    
    # creates the model w/ random weights
    # will talk about later on how to load pre-trained network
    model = LeNet4()

    # we will talk in the future about distributed training and inference
    # let us focus on single node/GPU for instance
    if args.gpu is not None and torch.cuda.is_available():
        # set GPU card to be used
        torch.cuda.set_device(args.gpu)
        # optimize model for best performance on the GPU
        cudnn.benchmark = True
        # transfer model parameters to GPU
        model = model.cuda(args.gpu)
        print("Use GPU: {} for training/inference".format(args.gpu))
    # else:
    #   model = model.cpu()
    
    train_loader, test_loader = loadData(args)
    
    # loss function
    criterion = nn.NLLLoss()
    # stochastic gradient descent optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # start the training epochs
    # train over all the train set
    # evaluates with the test set at the end of each epoch
    for epoch in range(1, args.epochs + 1):
        train_epoch_loss, train_epoch_acc = train(args, model, train_loader, criterion, optimizer, epoch)
        train_losses += [train_epoch_loss]
        train_accs += [train_epoch_acc]

        test_epoch_loss, test_epoch_acc =  test(args, model, criterion, test_loader)
        test_losses += [test_epoch_loss]
        test_accs += [test_epoch_acc]



if __name__ == '__main__':
    main()
