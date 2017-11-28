from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

from utils import map_label_to_target

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label ,dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            loss = self.criterion(output, target)
            total_loss += loss.data[0]
            loss.backward()
            if idx%self.args.batchsize==0 and idx>0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        total_loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1,dataset.num_classes+1)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            ltree,lsent,rtree,rsent,label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            loss = self.criterion(output, target)
            total_loss += loss.data[0]
            output = output.data.squeeze().cpu()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss/len(dataset), predictions
