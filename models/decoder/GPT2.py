from re import M
import numpy as np
import math
import pdb
from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from optuna import Trial
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data, HeteroData, Batch
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric

from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch_geometric.data.batch import Batch
from typing import Any, Dict, Literal, Optional, Tuple, Union, List
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from ..encoder.pretrained_encoder import DummyEncoder, PretrainedEncoder
import tutorials.train_trajectory_transformer.models.config as Config

from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from tutorials.train_trajectory_transformer.models.decoder.base_transformer_decoder import BaseTransformerDecoder, Block, CausalSelfAttention, BaseTransformerDecoderConfig
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Type

@dataclass
class GPT2Config(BaseTransformerDecoderConfig):
    pass
    
class GPT2(BaseTransformerDecoder):

    def __init__(self,
                config: Optional[GPT2Config] = None
                ):
        
        super().__init__()
        config = config or GPT2Config

    
    # def _build(self, data: Union[Batch, CommonRoadDataTemporal],trial: Union[None, Trial] = None) -> None:
    #     pass

    def forward(self, x: Tensor):
        """
            x : [ B x T x transition_dim ]
            values : [ B x 1 x 1 ]
        """
        #X, _=self.tok_emb(x)Union[Batch, CommonRoadDataTemporal]
        B, t, C = x.size()
        assert t <= self.sequence_length, "Cannot forward, model block size is exhausted."


        ## [ B x T x embedding_dim ]
        # forward the GPT model
        token_embeddings = x # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        ## [ B x T x embedding_dim ]
        pred = self.ln_f(x)

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        #x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        #logits = self.head(x)
        ## [ B x T' x (vocab_size + 1) ]
        #logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        #logits = logits[:,:t]

        return pred

@dataclass
class GPT2ProbDecoderConfig(BaseTransformerDecoderConfig):
    hidden_dim_multiplier: int = 4
    embd_pdrop: float = 0.1
    attn_drop: float = 0.1
    n_layers: int = 6

class GPT2ProbDecoder(BaseTransformerDecoder):

    def __init__(self,
                config: Optional[GPT2ProbDecoderConfig] = None,
                embd_dim: int = 12,
                n_head: int = 3,
                resid_pdrop: float = 0.1
                ):
        config = config or GPT2ProbDecoderConfig()
        super().__init__()
        self.config = config
        self.embd_dim = embd_dim
        self.tok_emb = nn.ModuleDict({})
        for i in range(Config.TRANSITION_DIM):
            self.tok_emb[str(i)]=nn.Embedding(Config.N_BINS, embd_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, Config.SEQUENCE_LENGTH,1, embd_dim))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(embd_dim)
        self.head1 = nn.Linear(embd_dim, config.hidden_dim_multiplier * embd_dim, bias=False)
        self.head2 = nn.Linear(config.hidden_dim_multiplier * embd_dim, Config.N_BINS, bias=False)
        self.blocks = nn.Sequential(*[Block(embd_dim,config.attn_drop,resid_pdrop,n_head) for _ in range(config.n_layers)])
    # def _build(self, data: Union[Batch, CommonRoadDataTemporal],trial: Union[None, Trial] = None) -> None:
    #     pass

    #def forward(self, x: Tensor, occupancy_latent: Tensor):
    def forward(self, x: Tensor, occupancy_embeddings:Tensor):
        """
            x : [ B x T x C ]
        """
        #X, _=self.tok_emb(x)Union[Batch, CommonRoadDataTemporal]
        B, t, C = x.size()
        assert t <= Config.SEQUENCE_LENGTH, "Cannot forward, model block size is exhausted."

        x=x.to(device=x.device,dtype=torch.int)
        ## [ B x T x transition_dim x embedding_dim ]
        # forward the GPT model
        graph_embeddings = torch.empty(B,t,C, self.embd_dim,device=x.device)

        for i in range(C):
            graph_embeddings[:,:,i,:] = self.tok_emb[str(i)](x[:,:,i]) # each index maps to a (learnable) vector
        
        ## [ 1 x T x 1 x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :, :] # each position maps to a (learnable) vector
        ## [ B x 1 x 1 x embedding_dim ]
        occupancy_embeddings = occupancy_embeddings.view(B,1,1,self.embd_dim)
        #another option is do concatenation like modify the mlp for occ 
        #and do occupancy_embeddings.view(B,T,C,whatever dim) and cat in dim -1
        
        ## [ B x T x transition_dim x embedding_dim ]
        x = self.drop(graph_embeddings + position_embeddings + occupancy_embeddings)

        x = self.blocks(x)
        ## [ B x T x C x embedding_dim ]
        x = self.ln_f(x)

        #[ B x T x transition_dim x n_bins ]
        #logits = self.head(x)
        logits = self.head2(self.head1(x))

        logits = logits[:,:t,:,:]
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        #x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        #logits = self.head(x)
        ## [ B x T' x (vocab_size + 1) ]
        #logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        #logits = logits[:,:t]

        return logits

@dataclass
class GPT2LatentEmbeddingConfig(BaseTransformerDecoderConfig):
    pass
    
class GPT2LatentEmbedding(BaseTransformerDecoder):

    def __init__(self):
        config: Optional[GPT2LatentEmbeddingConfig] = GPT2Config
        super().__init__()


    
    # def _build(self, data: Union[Batch, CommonRoadDataTemporal],trial: Union[None, Trial] = None) -> None:
    #     pass

    def forward(self, x: Tensor, occupancy_latent: Tensor):
        """
            x : [ B x T x transition_dim ]
            values : [ B x 1 x 1 ]
        """
        #X, _=self.tok_emb(x)Union[Batch, CommonRoadDataTemporal]
        B, t, C = x.size()
        assert t <= self.sequence_length, "Cannot forward, model block size is exhausted."


        ## [ B x T x embedding_dim ]
        # forward the GPT model
        graph_embeddings = x # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        occupancy_embeddings = torch.unsqueeze(occupancy_latent,dim=1)
        ## [ B x T x embedding_dim ]
        x = self.drop(graph_embeddings + position_embeddings + occupancy_embeddings)
        x = self.blocks(x)
        ## [ B x T x embedding_dim ]
        pred = self.ln_f(x)

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        #x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        #logits = self.head(x)
        ## [ B x T' x (vocab_size + 1) ]
        #logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        #logits = logits[:,:t]

        return pred