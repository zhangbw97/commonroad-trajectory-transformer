from dataclasses import dataclass
from optuna import Trial
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from typing import Any, Dict, Optional, Tuple, Union
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import tutorials.train_trajectory_transformer.models.config as Config
from dataclasses import dataclass


@dataclass
class BaseTransformerDecoderConfig(ABC):
    ...
class BaseTransformerDecoder(nn.Module, ABC):

    def __init__(self,
                config: Optional[BaseTransformerDecoderConfig] = None
                ):
        super().__init__()
        self.config = config
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @abstractmethod
    def forward(self, x: Tensor, occupancy_latent: Tensor) -> Tensor:
        """Computes prediction of vehicle states and actions in continuous space.

        Args:
            x: graph feature tensor provided from encoder
            occupancy_latent: lanelet occupancy latent of vehicle in its current lanelet
        Returns:
            A tuple of size [ B x T x C]
            the slice in last time step [ :, -1, :] corresponds to the prediction of vehicle states and actions at next time step
        """

    
class CausalSelfAttention(nn.Module):

    def __init__(self,
                embd_dim,
                attn_drop,
                resid_pdrop,
                n_head
                ):
        super().__init__()
        self.embd_dim=embd_dim
        self.observation_dim = Config.OBSERVATION_DIM
        self.action_dim=Config.ACTION_DIM
        self.sequence_length= Config.SEQUENCE_LENGTH
        # key, query, value projections for all heads
        self.key = nn.Linear(self.embd_dim, self.embd_dim)
        self.query = nn.Linear(self.embd_dim, self.embd_dim)
        self.value = nn.Linear(self.embd_dim, self.embd_dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_pdrop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.embd_dim, self.embd_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.register_buffer("mask", torch.tril(torch.ones(self.sequence_length, self.sequence_length))
                                .view(1, 1, 1, self.sequence_length, self.sequence_length))
        ## mask previous value estimates
        #joined_dim = self.observation_dim + self.action_dim + 2
        #self.mask.squeeze()[:,joined_dim-1::joined_dim] = 0
        ##
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        # B: batch size T: sequence length C: embedding dim
        B, T, C, E = x.size()

        ## calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # [ B x C x n_heads x T x head_dim ] head_dim=embedding_dim/n_heads
        k = self.key(x).view(B, T, C, self.n_head,  E// self.n_head).transpose(1, 3)
        q = self.query(x).view(B, T, C, self.n_head,  E// self.n_head).transpose(1, 3)
        v = self.value(x).view(B, T, C, self.n_head,  E// self.n_head).transpose(1, 3)

        ## causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) 
        ## [ B x T x n_heads x T ]

        #  (B C nh T ns) x (B C nh ns T) -> (B T nh T T)
        # [ B x T x n_heads x T x T ]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))


        att = att.masked_fill(self.mask[:,:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        ## [ B x n_heads x T x C x head_dim]
        y = att @ v # (B, C, nh, T, T) x (B, C, nh, T, hs) -> (B, C, nh, T, hs)
        ## [ B x T x embedding_dim ]
        #y = y.transpose(1, 2).contiguous().view(B, T, C, E) # re-assemble all head outputs side by side
        y = y.permute(0,3,1,2,4).contiguous().view(B,T,C,E)
        # output projection [B, T, C]
        y = self.resid_pdrop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self,
                embd_dim,
                attn_drop,
                resid_pdrop,
                n_head
                ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.attn = CausalSelfAttention(embd_dim,
                                        attn_drop,
                                        resid_pdrop,
                                        n_head)
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        #residual connection
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
