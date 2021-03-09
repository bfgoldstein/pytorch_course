import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import *


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
    

def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    # using meters to track batch execution time
    batch_time = AverageMeter()
    
    # using meters to track accuracies
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output = model(input)
            
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # update accuracies meters
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
            # access the topN scores and predictions
            scores, predictions = topN(output, target, topk=(1,5))
                        
            # update batch time meter
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Prediction Run: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, top1=top1, top5=top5))

        print('Prediction Run * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return


# load the MNIST dataset into a dataloader object
def loadData(args):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
                    batch_size=args.test_batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(args.gpu is not None))
    
    return test_loader


def createModel(args):
    # models = torchvision.models
        
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](pretrained=True)
        model = LeNet4()
        if args.checkpoint_file is not None:
            checkpoint = torch.load(args.checkpoint_file, map_location="cuda:" + str(args.gpu) if args.gpu is not None else 'cpu')
            model.load_state_dict(checkpoint)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = LeNet4()    
    
    return model


def main():
    parser_obj = parser.getParser()
    args = parser_obj.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True
    
    model = createModel(args)
    
    # we will talk in the future about distributed training and inference
    # let us focus on single node/GPU for instance
    if args.gpu is not None and torch.cuda.is_available():
        print("Use GPU: {} for inference".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    # else:
    #     model.cpu()
    
    test_loader = loadData(args)
    
    validate(test_loader, model, args)
    
    
if __name__ == '__main__':
    main()