
import torch

class FramewiseHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(FramewiseHead, self).__init__()
        self.left_fc = torch.nn.Linear(768, num_classes)
        self.right_fc = torch.nn.Linear(768, num_classes)

    
    def forward(self, x):
        bs = x.size(0)
        x = x[:, 1:, :] # remove the first token, which is the CLS token
        x = x.reshape(bs, 16, 14, 14, 768) # FIXME: hardcoded shape
        x = x.mean(dim=[2, 3])
        left = self.left_fc(x)
        right = self.right_fc(x)

        return torch.cat([left, right], dim=0)
