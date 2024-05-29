import torch, torch.nn as nn
from .ABMIL import ABMIL
from .DTFDMIL import DTFDMIL
from .TransMIL import TransMIL
from .fusion import Concatenate, AttentionPooling

EXTRACTOR_OUT_DIM = {"conch": 512, "gigapath": 1536, "hipt": 384, "resnet": 2048}


def get_fusion(args, in_dim, **kwargs):
    name = args.fusion
    if name == "concat":
        return Concatenate()
    elif name == "attn_pool":
        return AttentionPooling(in_dim, args.attn_out_dim, args.attn_shared_dim)
    elif name is None:
        return None
    else:
        raise ValueError("name of fusion ({}) not available".format(name))


def get_mil_aggregator(args, in_dim, **kwargs):
    name = args.mil_aggregator

    if name == "abmil":
        return ABMIL(in_dim, args.num_cls)
    elif name == "transmil":
        return TransMIL(in_dim, args.num_cls)
    elif name == "dtfdmil":
        return DTFDMIL(in_dim, args.num_cls)
    else:
        raise ValueError("name of MIL aggregator ({}) not available".format(name))


class FusionMIL(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        # feature extractor(s)
        self.feature_extractor = args.extractor.split(",")
        self.in_dim = [EXTRACTOR_OUT_DIM[i] for i in self.feature_extractor]

        # features fusion
        if args.fusion is None or args.fusion == "concat":
            self.out_dim = sum(self.in_dim)
        else:
            self.out_dim = args.attn_out_dim
        
        if len(self.feature_extractor) > 1:
            self.fusion = get_fusion(args, self.in_dim)
            assert not self.fusion is None, "If several feature extractors are used, then a fusion strategy must be selected"
        else:
            self.fusion = None

        # MIL aggregator
        self.mil_aggregator = get_mil_aggregator(args, self.out_dim)

        assert hasattr(self.mil_aggregator, "calculate_loss"), "MIL aggregator must have callable function `calculate_loss` to calculate loss"

    def forward(self, x, return_attn=False, **kwargs):
        """
        args:
            x (torch.Tensor or list of torch.Tensor): each tensor is a batch of 1 slide of size (1,N,C)

        return logits of classification layer
        """

        def check_tensor(t):
            if len(t.shape) == 3:
                if t.shape[0] > 1:
                    raise ValueError('input(s) must have shape like (1,N,C) but received {}'.format(t.shape))
            else:
                raise ValueError('input(s) must have shape like (1,N,C) but received {}'.format(t.shape))
            
            return t
        
        if isinstance(x, torch.Tensor):
            x = check_tensor(x)
        else:
            x = [check_tensor(t) for t in x]

        if len(self.feature_extractor) > 1:
            x = self.fusion(x)
        
        if return_attn:
            out, attn = self.mil_aggregator(x, return_attn=True)
            return out, attn
        else:
            out = self.mil_aggregator(x, return_attn=False)
            return out

    
    def calculate_loss(self, y_hat, y, loss_fn, **kwargs):
        """
        Calculate loss based on MIL aggregator

        args:
            y_hat (torch.Tensor): output of MIL aggregator
            y (torch.Tensor): ground-truth as class index
            loss_fn (function): function that computes the loss

        return the loss to use for backward
        """

        return self.mil_aggregator.calculate_loss(y_hat, y, loss_fn)
