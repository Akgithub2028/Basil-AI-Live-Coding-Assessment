"""One prior: local temporal filters shared across EEG electrodes.

The fixed scattered-time control has exactly the same learned tensors/shapes.
No modification of models.py or its vanilla transformer is needed.
"""
import numpy as np
import torch
from torch import nn


class YourModel(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2, n_filters=8,
                 kernel=32, stride=16, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.n_chan, self.n_time = n_chan, n_time
        self.n_filters, self.kernel, self.stride = n_filters, kernel, stride
        self.n_tokens = (n_time-kernel)//stride+1
        if self.n_tokens < 1:
            raise ValueError("Input must be at least one temporal window long")
        self.register_buffer("time_index",torch.arange(n_time))
        # Reshaping electrodes into the batch axis forces temporal weight sharing.
        self.temporal = nn.Conv1d(1,n_filters,kernel_size=kernel,stride=stride,bias=False)
        self.proj = nn.Linear(n_chan*n_filters,d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos = nn.Parameter(torch.zeros(1,self.n_tokens+1,d_model))
        nn.init.trunc_normal_(self.pos,std=0.02)
        nn.init.trunc_normal_(self.cls,std=0.02)
        layer = nn.TransformerEncoderLayer(d_model,n_heads,dim_feedforward=2*d_model,
                     dropout=p,batch_first=True,activation="gelu")
        self.encoder = nn.TransformerEncoder(layer,depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,n_classes)

    def tokenize(self, x):
        b,c,t = x.shape
        if (c,t) != (self.n_chan,self.n_time):
            raise ValueError(f"Expected {(self.n_chan,self.n_time)}, got {(c,t)}")
        ordered = x.index_select(-1,self.time_index)
        features = self.temporal(ordered.reshape(b*c,1,t))
        features = features.reshape(b,c,self.n_filters,self.n_tokens)
        return self.proj(features.permute(0,3,1,2).reshape(b,self.n_tokens,c*self.n_filters))

    def forward(self,x,return_embed=False):
        tokens = self.tokenize(x)
        z = torch.cat([self.cls.expand(x.shape[0],-1,-1),tokens],dim=1)+self.pos
        embedding = self.norm(self.encoder(z)[:,0])
        logits = self.head(embedding)
        return (logits,embedding) if return_embed else logits


class ScatteredTimeControl(YourModel):
    def __init__(self,n_chan,n_time,**kwargs):
        super().__init__(n_chan,n_time,**kwargs)
        used = (self.n_tokens-1)*self.stride+self.kernel
        index = np.arange(n_time)
        index[:used] = np.random.default_rng(2026).permutation(used)
        self.time_index.copy_(torch.as_tensor(index))
