
from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILENAME
from tutorials.train_geometric_model.models.occupancy.occupancy_model import OccupancyModel
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from abc import ABC, abstractmethod
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
#from tutorials.train_geometric_model.models.occupancy.conv_layers import L2LLConvLayer, V2LConvLayer
from commonroad_geometric.learning.geometric.components.mlp import MLP
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing, HeteroConv
import tutorials.train_trajectory_transformer.models.config as Config
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
#model_filepath='tutorials/train_trajectory_transformer/output/model/pretrained/model.pt'
#model_filepath='tutorials/train_geometric_model/models/occupancy/pretrained/model.pt'
model_filepath='tutorials/train_trajectory_transformer/pretrained_models/occ_model_highd.pt'

@dataclass
class BaseGraphEncoderConfig(ABC):
    ...
    
class BaseGraphEncoder(MessagePassing):
    def __init__(self,
                config: Optional[BaseGraphEncoderConfig] = None
                ):
        super(MessagePassing, self).__init__()
        
        self._model = OccupancyModel()
        self.pretrained_model=self._model.load(model_filepath, device='cuda')
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.config = config
        self.apply(self._init_weights)

    @abstractmethod
    def build(
        self,
        data: CommonRoadDataTemporal
    ) -> None:
        ...
 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @abstractmethod
    def forward(self, data_temporal:CommonRoadDataTemporal) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        get the tensor of each vehicle, obtain the occupancy embedding of the current lanelet of vehicle, velocity and orientation as state, acceleration and yaw_rate as action,
        only accept the trajectory as output when the vehicle exists for 40 timesteps.

        3 modes need to be enabled: 1. training mode 2. validation mode 3. planning(rendering) mode
        planning(rendering) mode has different dimension as 1 and 2 mode, since it takes different length of sequence as input
        Args:
            data (CommonRoadDataTemporal)

        Returns:
            Tensor: input to transformer
            [pos_x, pos_y, velocity, orientation,lanelet_arclength_abs, lanelet_arclength_rel,dist_left_bound,dist_right_bound,
            lanelet_lateral_error, heading error, acceleration, yaw rate]
            lanelet occupancy embedding can either in outputed in seperate tensor or together.
        """