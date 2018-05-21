import torch

A = torch.zeros((3,5))
B = torch.zeros((3,2))
idxs = torch.LongTensor([0, 1])

# A.index_copy_(1,idxs,B)

torch.transpose(A,1,0).index_copy_(0,idxs,torch.transpose(B,1,0))