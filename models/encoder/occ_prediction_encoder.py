

from dataclasses import dataclass
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
@dataclass
class OccPredictionEncoderConfig(BaseGraphEncoderConfig):
    activation_cls: Type[nn.Module] = GELU


class OccPredictionEncoder(BaseGraphEncoder):
        def __init__(self):
            config: Optional[OccPredictionEncoderConfig] = OccPredictionEncoderConfig
            super().__init__()
            self.ln_f = nn.LayerNorm(Config.TRANSITION_DIM)
            self.bn_f = nn.BatchNorm1d(Config.TRANSITION_DIM)


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

                [delta(lanelet_arclength_abs), delta(lanelet_lateral_error), heading_error(instead of orientation in global frame), velocity, acceleration, yaw_rate], 
                [lanelet_arclength_rel,lanelet_lateral_error] are related to lanelet occupancy encoding

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
                # prediction with history
                if self.training is False and data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] == Config.HISTORY_LENGTH: 
                    ids_accepted.append(i)

                #if self.training:
                #In training or validation stage, only accept full sequence length
                elif data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0]==Config.SEQUENCE_LENGTH:
                    ids_accepted.append(i)

                # if data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] != 0:
                #     ids_accepted.append(i)

            ################### graph embedding ###########################

            x_t=torch.cat([torch.unsqueeze(data_temporal.get_node_features_temporal_sequence_for_vehicle(i),0)  for i in ids_accepted], dim=0).to(device=data_temporal.vehicle.id.device)
            x_log_delta_lanelet_arclength_abs=torch.cat([torch.unsqueeze(get_log_delta_lanelet_arclength_abs(data_temporal,i),0)  for i in ids_accepted], dim=0)
            x_delta_lanelet_lateral_error=torch.cat([torch.unsqueeze(get_delta_lanelet_lateral_error(data_temporal,i),0)  for i in ids_accepted], dim=0)
            #[accepted vehicle(trajectories) num, sequence length, feature_dim]
            #index=torch.tensor([11,0,25,26,1,3]).to(device=data_temporal.vehicle.id.device)
            index=torch.tensor([11,0,1,3,7,10]).to(device=data_temporal.vehicle.id.device)
            x_t_accepted= torch.cat([x_log_delta_lanelet_arclength_abs.unsqueeze(dim=2),x_delta_lanelet_lateral_error.unsqueeze(dim=2),\
            x_t.index_select(-1,index)],dim=2).to(device=data_temporal.vehicle.id.device)

            ################### occupancy latent embedding###########################
            self.pretrained_model.eval()
            time_step_min=data_temporal.time_step[0]
            time_step_max=data_temporal.time_step[-1]
            T =time_step_max-time_step_min+1
            out_occupancy = torch.empty(len(ids_accepted),0,Config.OCC_EMBEDDING_DIM,device=data_temporal.vehicle.id.device)
            for t in range(T):
                data=data_temporal[t]
                # [Lanelet_num x occ_embd_dim on each lanelet]
                occupancy_encodings=self.pretrained_model.encode(data)

                lanelet_encoding=torch.empty(0,Config.OCC_EMBEDDING_DIM,device=data_temporal.vehicle.id.device)

                counter=0
                for i in data.vehicle.id:    
                    if i in ids_accepted:
                        
                        #get the lanelet occupancy encoding between accepted vehicle and its current lanelet
                        lanelet_encoding=torch.cat((lanelet_encoding,\
                        torch.unsqueeze((occupancy_encodings[data.vehicle_to_lanelet.edge_index[1,counter]]),dim=0)),dim=0)

                    counter = counter + 1

                out_occupancy = torch.cat((out_occupancy,torch.unsqueeze(lanelet_encoding,dim=1)),dim=1)

            # Attention based vehicle to lanelet occupancy embedding
            # out_occupancy = self.mlp(lanelet_encoding)
            # out_occupancy = self.bn_f(out_occupancy) 

            out = torch.cat((x_t_accepted,out_occupancy),dim=2)

            if out.shape[1]==1:
                out_x = out
                out_y = out
            else:
                out_x=out[:,:-1,:]
                out_y=out[:,1:,:]
            return out_x, out_y

