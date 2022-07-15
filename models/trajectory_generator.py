from dataclasses import dataclass
from unicodedata import decimal
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
from tutorials.train_geometric_model.models.occupancy.occupancy_model import OccupancyModel
from tutorials.train_trajectory_transformer.models.decoder.GPT2 import GPT2, GPT2LatentEmbedding, GPT2Config, GPT2LatentEmbeddingConfig, GPT2ProbDecoder, GPT2ProbDecoderConfig
from tutorials.train_trajectory_transformer.models.decoder.base_transformer_decoder import BaseTransformerDecoder, BaseTransformerDecoderConfig
from tutorials.train_trajectory_transformer.models.encoder.base_graph_encoder import BaseGraphEncoder, BaseGraphEncoderConfig
from tutorials.train_trajectory_transformer.models.encoder.mlp_occ_encoder import MLPOccEncoder, MLPOccEncoderConfig
from tutorials.train_trajectory_transformer.models.encoder.occ_prediction_encoder import OccPredictionEncoder, OccPredictionEncoderConfig
from tutorials.train_trajectory_transformer.models.encoder.pretrained_encoder import PretrainedEncoder, DummyEncoder
from commonroad_geometric.rendering.plugins import RenderLaneletNetworkPlugin, RenderTrafficGraphPlugin, RenderObstaclesPlugin, \
    RenderEgoVehiclePlugin, RenderPlanningProblemSetPlugin, RenderEgoVehicleInputPlugin, RenderEgoVehicleCloseupPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_temporal_plugin import RenderObstaclesTemporalPlugin
from tutorials.train_trajectory_transformer.models.render_trajectory_prediction_plugin import RenderTrajectoryPredictionPlugin
import tutorials.train_trajectory_transformer.models.config as Config
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin

@dataclass
class TrajectoryGeneratorConfig():

    encoder_cls: Type[BaseGraphEncoder] = MLPOccEncoder
    encoder_config: Optional[BaseGraphEncoderConfig] = MLPOccEncoderConfig
    decoder_cls: Type[BaseTransformerDecoder] = GPT2ProbDecoder
    decoder_config: Optional[BaseTransformerDecoderConfig] = GPT2ProbDecoderConfig
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.99, 0.995)
    resid_pdrop: float = 0.1
    weight_decay: float = 0.1
    action_weight: int = 2
    observation_weight: int = 1
    #embd_dim: int = 12
    embd_dim_and_n_head: Tuple[int,int] = (12,3)
class TrajectoryGenerator(BaseGeometric):
    def __init__(self):
        super().__init__()
        config: Type[TrajectoryGeneratorConfig] = TrajectoryGeneratorConfig
        self.config = config
        self.encoder_cls=config.encoder_cls
        self.decoder_cls = config.decoder_cls

        self.encoder = self.encoder_cls(
            embd_dim=config.embd_dim_and_n_head[0],
            resid_pdrop = config.resid_pdrop,
            config=config.encoder_config
        )
        self.decoder = self.config.decoder_cls(
            embd_dim = config.embd_dim_and_n_head[0],
            n_head = config.embd_dim_and_n_head[1],
            resid_pdrop = config.resid_pdrop,
            config=config.decoder_config
        )
        # for wandb export
        config.encoder_config = self.encoder.config
        config.decoder_config = self.decoder.config

        self.sequence_length = Config.SEQUENCE_LENGTH
        self.observation_dim = Config.OBSERVATION_DIM
        self.occ_embedding_dim = Config.OCC_EMBEDDING_DIM
        self.action_dim = Config.ACTION_DIM
        self.transition_dim = Config.TRANSITION_DIM
        self.action_weight = config.action_weight
        self.observation_weight = config.observation_weight
        self.embedding_dim = config.embd_dim_and_n_head[0]

    def _build(self, data: Union[Batch, CommonRoadDataTemporal],trial: Union[None, Trial] = None) -> None:
        pass

    def forward(self,
                data: Union[Batch, CommonRoadDataTemporal],    
                validation: bool = False) -> Tensor:
            
        if isinstance(self.encoder, MLPOccEncoder):
            X, Y, occ_embedding = self.encoder(data)  
        elif isinstance(self.encoder, OccPredictionEncoder):
            X, Y= self.encoder(data) 

        # if isinstance(self.decoder,GPT2):
        #     pred = self.decoder(X) 
        if isinstance(self.decoder,GPT2ProbDecoder):
            logits = self.decoder(X,occ_embedding) 
        # elif isinstance(self.decoder, GPT2LatentEmbedding):
        #     pred = self.decoder(X, Occupancy_latent)
        #[B x T x C x V]
        return logits

    def compute_loss(
        self,
        out: Tensor,
        data: CommonRoadDataTemporal,
        epoch: int,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any], Dict[str, Tensor]]:  
        # if we are given some desired targets also calculate the loss
        if isinstance(self.encoder, PretrainedEncoder):
            _,targets = self.encoder(data)
            
        elif isinstance(self.encoder, MLPOccEncoder):
            #_,targets, Occupancy_latent = self.encoder(data)
            _,targets,_ = self.encoder(data)

        elif isinstance(self.encoder, OccPredictionEncoder):
            _,targets = self.encoder(data)

        B,T,C,V = out.size()
        if targets is not None:
            loss =  F.cross_entropy(out.reshape(-1, out.size(-1)), targets.view(-1), reduction='none')
            if isinstance(self.encoder, MLPOccEncoder):
                if self.action_weight != 1  or self.observation_weight != 1:
                    #### make weights
                    #n_states = int(np.ceil(T / self.transition_dim))
                    #state_weights=torch.ones(self.observation_dim, device=out.device)
                    state_weights=torch.tensor([self.observation_weight for _ in range(self.observation_dim)],device=out.device,dtype=torch.int64)
                    weights = torch.cat((
                        state_weights,
                        torch.ones(self.action_dim, device=out.device,dtype=torch.int64) * self.action_weight,
                        #torch.ones(1, device=out.device) * self.reward_weight,
                        #torch.ones(1, device=out.device) * self.value_weight,
                    ))
                    ## [ t ]
                    #weights = weights.repeat(self.sequence_length)
                    ## [ b x t ]
                    weights = weights.repeat(B,T,1)
                    #out = out * weights
                    #targets = targets * weights
                else:
                    weights = torch.ones((B,T,C),device = out.device)
            loss = loss * weights.view(-1)
            loss = loss.mean()
            info = dict(
                out_max=out.max().item(),
                out_min=out.min().item(),
                out_std=out.std().item(),
                )

        else:
            loss = None
        
        return loss, info
        
    from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService
    def configure_optimizer(
        self,         
        trial: Any = None, 
        optimizer_service: BaseOptimizerService = None
    ) -> Optimizer:

        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,nn.Embedding )
        blacklist_weight_modules = (nn.LayerNorm, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayedDataset4VALIDAT
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.find('pos_emb') != -1 and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        #no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        return optimizer
        # return torch.optim.Adam(
        #     self.parameters(), 
        #     lr=Config.LEARNING_RATE,
        #     betas=(0.9, 0.995)
        # )




    def configure_visualizer(self) -> Optional[List[BaseRendererPlugin]]:

        return [
        RenderLaneletNetworkPlugin(render_id=False, enable_ego_rendering=False, from_graph=True),
        #RenderPlanningProblemSetPlugin(render_trajectory=True),
        RenderTrafficGraphPlugin(),
        RenderObstaclesPlugin(),
        # RenderObstaclesTemporalPlugin(
        #     max_prev_time_steps=10,
        #     randomize_color=True,
        # ),
        RenderTrajectoryPredictionPlugin(),
        #RenderEgoVehiclePlugin(),
        #RenderEgoVehicleInputPlugin(),
        #RenderEgoVehicleCloseupPlugin(),
        # RenderObstaclesEgoFramePlugin()
        # RenderVehicleToLaneletEdges()
    ]