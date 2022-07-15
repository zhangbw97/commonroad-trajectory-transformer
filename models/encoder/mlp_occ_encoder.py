

from dataclasses import dataclass
import os
from tutorials.train_geometric_model.models.occupancy.occupancy_model import OccupancyModel
from commonroad_geometric.dataset.commonroad_data import CommonRoadData

from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
#from tutorials.train_geometric_model.models.occupancy.conv_layers import L2LLConvLayer, V2LConvLayer
from commonroad_geometric.learning.geometric.components.mlp import MLP
from torch import Tensor
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, HeteroConv
import tutorials.train_trajectory_transformer.models.config as Config
from tutorials.train_trajectory_transformer.models.encoder.base_graph_encoder import BaseGraphEncoder, BaseGraphEncoderConfig
from typing import Any, Dict, Optional, Tuple, Union, Type
from torch.nn import ReLU, Tanh, PReLU, LeakyReLU, Hardtanh, GELU
from tutorials.train_trajectory_transformer.helper import  get_delta_lanelet_arclength_rel, get_delta_lanelet_lateral_error, get_delta_lanelet_arclength_abs, get_log_delta_lanelet_arclength_abs
from tutorials.train_trajectory_transformer.temporal_experiment import TemporalGeometricExperiment
@dataclass
class MLPOccEncoderConfig(BaseGraphEncoderConfig):
    activation_cls: Type[nn.Module] = GELU
    hidden_dim_multiplier: int = 4 
class MLPOccEncoder(BaseGraphEncoder):
        def __init__(self,
                    config: Optional[MLPOccEncoderConfig] = None,
                    embd_dim: int = 12,
                    resid_pdrop: float = 0.1
                    ):
            config = config or MLPOccEncoderConfig
            super().__init__()
            self.config = config
            self.ln_f = nn.LayerNorm(embd_dim)
            self.bn_f = nn.BatchNorm1d(embd_dim)
            self.mlp = nn.Sequential(
                nn.Linear(3 * (Config.OCC_EMBEDDING_DIM + 1), config.hidden_dim_multiplier * embd_dim),
                nn.BatchNorm1d( config.hidden_dim_multiplier * embd_dim),
                config.activation_cls(),
                nn.Linear( config.hidden_dim_multiplier * embd_dim, embd_dim),
                nn.Dropout(resid_pdrop),
            )

        def build(self, data:CommonRoadDataTemporal) -> None:
            pass

        def forward(self, data_temporal:CommonRoadDataTemporal) -> Tensor:
            """
            get the tensor of each vehicle, obtain the occupancy embedding of the current lanelet of vehicle, velocity and orientation as state, acceleration and yaw_rate as action,
            only accept the trajectory as output when the vehicle exists for 40 timesteps.

            3 modes need to be enabled: 1. training mode 2. validation mode 3. planning(rendering) mode
            planning(rendering) mode has different dimension as 1 and 2 mode, since it takes different length of sequence as input
            Args:
                data (CommonRoadDataTemporal)

            Returns:
                Tensor: input to transformer decoder [B x (T - 1) x C]

                [delta(lanelet_arclength_abs), delta(lanelet_lateral_error), heading_error(instead of orientation in global frame), velocity], 
                for traffic prediction, following are not needed 
                [goal_distance_long_noego, goal_distance_lat_noego, acceleration, yaw_rate]

                out_y: training target [B x (T - 1) x C]

                lanelet occupancy embedding: Tensor [B x C] 
            """

            min_id=torch.min(data_temporal.vehicle.id)
            max_id=torch.max(data_temporal.vehicle.id)
            ids=list(range(min_id,max_id+1))

            ################ vehicle filter ###########################
            #accept vehicles that exist for 40 time steps
            ids_accepted=[]

            for i in ids:
                # If only 1 time step provided, it's planning(rendering) mode
                if self.training is False and data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] == 1: 
                    ids_accepted.append(i)

                #if self.training:
                #In training or validation stage, only accept full sequence length
                elif data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0]==Config.SEQUENCE_LENGTH:
                    ids_accepted.append(i)

                # if data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] != 0:
                #     ids_accepted.append(i)

            ################### graph embedding ###########################

            x_t=torch.cat([torch.unsqueeze(data_temporal.get_node_features_temporal_sequence_for_vehicle(i),0)  for i in ids_accepted], dim=0).to(device=data_temporal.vehicle.id.device)
            #x_log_delta_lanelet_arclength_abs=torch.cat([torch.unsqueeze(get_log_delta_lanelet_arclength_abs(data_temporal,i),0)  for i in ids_accepted], dim=0)
            #x_delta_lanelet_lateral_error=torch.cat([torch.unsqueeze(get_delta_lanelet_lateral_error(data_temporal,i),0)  for i in ids_accepted], dim=0)
    
            #[accepted vehicle(trajectories) num, sequence length, feature_dim]
            index=torch.tensor([11,0,25,26,1,3]).to(device=data_temporal.vehicle.id.device)
            #index=torch.tensor([11,0]).to(device=data_temporal.vehicle.id.device)
            x_t_accepted= torch.cat([data_temporal.v2l.delta_lanelet_arclength_abs.view(len(ids_accepted),-1,1),data_temporal.v2l.delta_lanelet_lateral_error.view(len(ids_accepted),-1,1),\
            x_t.index_select(-1,index)],dim=2).to(device=data_temporal.vehicle.id.device)
  
            # put feature dimension to the second dimension for batchnorm1d
            # out = x_t_accepted.permute(0,2,1)
            # out = self.bn_f(out)
            # out = out.permute(0,2,1)
            out = x_t_accepted
            if out.shape[1]==1:
                source = out
                target = out
                B, _, C = out.shape
            else:
                source=out[:,:-1,:]
                target=out[:,1:,:].to(dtype=torch.int64)
                #TODO
                # apply one hot to target
                B, t, C = target.shape 
                # target = torch.zeros( (B, t, C, Config.N_BINS), device=out.device, dtype=torch.int64)
                # for i in range (B):
                #     for j in range(t):
                #         for k in range(C):
                #             target[i,j,k,int(out_pred[i,j,k].item())] = 1
            ################### occupancy latent embedding###########################
            # self.pretrained_model.eval()
            # time_step_min=data_temporal.time_step[0]
            # time_step_max=data_temporal.time_step[-1]
            # lanelet_encoding_temporal_lst= []

            # # Obtain the occupancy latent of the first time step and use it for predicting all time_steps
            data=data_temporal[0]
            # occupancy_encodings=self.pretrained_model.encode(data)

            # z_lanelet = occupancy_encodings[0] if isinstance(occupancy_encodings, tuple) else occupancy_encodings
            #[B x 3 x 32]
            route_encoding=torch.empty((0, 3, Config.OCC_EMBEDDING_DIM+1), device=data_temporal.vehicle.id.device)

            # counter=0
            for vehicle_idx, vehicle_id in enumerate(data.vehicle.id):    
                if vehicle_id in ids_accepted:
                    vehicle_occ_encoding = torch.unsqueeze(data_temporal.v.occ_encoding[vehicle_idx],dim=0)
            #         #get the lanelet occupancy encoding between accepted vehicle and its current lanelet
            #         lanelet_encoding=torch.cat((lanelet_encoding,\
            #         torch.unsqueeze((occupancy_encodings[data.vehicle_to_lanelet.edge_index[1,counter]]),dim=0)),dim=0)

            #     counter = counter + 1
                    route_encoding = torch.cat((route_encoding,vehicle_occ_encoding),dim=0)

            out_occupancy = self.mlp(route_encoding.view(B, 3 * (Config.OCC_EMBEDDING_DIM+1)))
            out_occupancy = self.bn_f(out_occupancy) 

            return source, target, out_occupancy

