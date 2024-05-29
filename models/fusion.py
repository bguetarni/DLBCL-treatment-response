import torch, torch.nn as nn
import torch.nn.functional as F

class Concatenate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        args:
            x (iterable): tuple/list of torch.Tensor
        """

        return torch.cat(x, dim=-1)
    

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, out_dim, shared_dim=None, **kwargs):
        """
        The attention pooling is based on [Attention-based Deep Multiple Instance Learning] paper
        The dimension in the paper are represented below inside brackets

        We adapt the attention pooling to consider input feature vectors of different dimensions.

        args:
            in_dim (tuple): tuple of ints indicating the dimension of each feature vector
            out_dim (int): dimension of output feature vector [M]
            shared_dim (int): dimension in the shared space before attention [L]
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shared_dim = out_dim if shared_dim is None else shared_dim
        
        # projection layer to project feature vectors to shared dimension
        self.proj = nn.ModuleList([nn.Linear(i, out_dim) for i in self.in_dim])

        # attention parameter V        
        self.attention_V = nn.Sequential(
            nn.Linear(self.out_dim, self.shared_dim),
            nn.Tanh()
        )

        # attention parameter U        
        self.attention_U = nn.Sequential(
            nn.Linear(self.out_dim, self.shared_dim),
            nn.Sigmoid()
        )

        # attention parameter w        
        self.attention_w = nn.Linear(self.shared_dim, 1)

    def forward(self, x):
        """
        args:
            x (iterable of torch.Tensor) iterable of feature vectors, each of shape (N,C_i)
        """

        # project feature vectors to shared dimension
        x = [proj(i) for proj, i in zip(self.proj, x)]
        x = torch.stack(x, dim=-2)
        
        # attention weights
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_w(A_V * A_U)
        A = torch.transpose(A, -1, -2)
        A = F.softmax(A, dim=1)

        # weighted sum
        Z = torch.matmul(A, x)

        return Z.squeeze(dim=-2)
