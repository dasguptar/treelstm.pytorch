import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, sparsity):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.emb = nn.Embedding(vocab_size,in_dim,
                                padding_idx=Constants.PAD,
                                sparse=sparsity)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fx = nn.Linear(self.in_dim,self.mem_dim)
        self.fh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h,1),0)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs),1)
        f = F.torch.cat([self.fh(child_hi)+fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f,1)
        fc = F.torch.squeeze(F.torch.mul(f,child_c),1)

        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o, F.tanh(c))

        return c,h

    def forward(self, tree, inputs):
        # add singleton dimension for future call to node_forward
        embs = F.torch.unsqueeze(self.emb(inputs),1)
        for idx in xrange(tree.num_children):
            _ = self.forward(tree.children[idx], inputs)
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx], child_c, child_h)
        return tree.state

    def get_child_states(self, tree):
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in xrange(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h

# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2*self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = F.torch.mul(lvec, rvec)
        abs_dist = F.torch.abs(F.torch.add(lvec,-rvec))
        vec_dist = F.torch.cat((mult_dist, abs_dist),1)
        out = F.sigmoid(self.wh(vec_dist))
        # out = F.sigmoid(out)
        out = F.log_softmax(self.wp(out))
        return out

# puttinh the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity):
        super(SimilarityTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.childsumtreelstm = ChildSumTreeLSTM(cuda, vocab_size, in_dim, mem_dim, sparsity)
        self.similarity = Similarity(cuda, mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output
