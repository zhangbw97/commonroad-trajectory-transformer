

from commonroad_geometric.dataset.commonroad_data import CommonRoadData

from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
#from tutorials.train_geometric_model.models.occupancy.conv_layers import L2LLConvLayer, V2LConvLayer
from commonroad_geometric.learning.geometric.components.mlp import MLP
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing, HeteroConv
import tutorials.train_trajectory_transformer.models.config as Config
from tutorials.train_trajectory_transformer.models.encoder.base_graph_encoder import BaseGraphEncoder
class DummyEncoder(BaseGraphEncoder):
        def __init__(self):
            super(MessagePassing, self).__init__()

    
        def build(self, data:CommonRoadDataTemporal) -> None:
            pass    
        
        def forward(self, data:CommonRoadDataTemporal) -> Tensor:
            """
            get the tensor of each vehicle, obtain the velocity and orientation as state, acceleration and yaw_rate as action,
            only accept the trajectory as output when the vehicle exists for 40 timesteps.

            Args:
                data (CommonRoadDataTemporal)

            Returns:
                Tensor: input to transformer
            """
            min_id=torch.min(data.vehicle.id)
            max_id=torch.max(data.vehicle.id)
            ids=list(range(min_id,max_id+1))
            ids_accepted=[]
            for i in ids:
                if data.get_node_features_temporal_sequence_for_vehicle(i).shape[0]==Config.SEQUENCE_LENGTH:
                    ids_accepted.append(i)
            x_t=torch.cat([torch.unsqueeze(data.get_node_features_temporal_sequence_for_vehicle(i),0)  for i in ids_accepted], dim=0)
            x_t_accepted=x_t.index_select(-1,torch.tensor([0,4,1,5]).to(device=data.vehicle.id.device))
  
            #[accepted vehicle(trajectories) num, sequence length, feature_dim]
            return x_t_accepted



class PretrainedEncoder(BaseGraphEncoder):
        def __init__(self):
            super().__init__()
            

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
                Tensor: input to transformer
                [pos_x, pos_y, velocity, orientation,lanelet_arclength_abs, lanelet_arclength_rel,dist_left_bound,dist_right_bound,
                lanelet_lateral_error, heading error, lanelet occupancy embedding, acceleration, yaw rate]
            """

            min_id=torch.min(data_temporal.vehicle.id)
            max_id=torch.max(data_temporal.vehicle.id)
            ids=list(range(min_id,max_id+1))

            #accept vehicles that exist for 40 time steps
            ids_accepted=[]

            for i in ids:
                # If only 1 time step provided, it's planning(rendering) mode
                if self.training is False and data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] == 1: 
                    ids_accepted.append(i)
                    continue
                #if self.training:
                #In training or validation stage, only accept full sequence length
                elif data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0]==Config.SEQUENCE_LENGTH:
                    ids_accepted.append(i)

                # if data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] != 0:
                #     ids_accepted.append(i)

            x_t=torch.cat([torch.unsqueeze(data_temporal.get_node_features_temporal_sequence_for_vehicle(i),0)  for i in ids_accepted], dim=0).to(device=data_temporal.vehicle.id.device)
            x_pos=torch.cat([torch.unsqueeze(get_pos_temporal_sequence_for_vehicle(data_temporal,i),0)  for i in ids_accepted], dim=0).to(device=data_temporal.vehicle.id.device)

            #[accepted vehicle(trajectories) num, sequence length, feature_dim]
            index=torch.tensor([0,2,6,7,8,9,10,11,1,3]).to(device=data_temporal.vehicle.id.device)
            x_t_accepted=torch.cat([x_pos, x_t.index_select(-1,index)], dim=2).to(device=data_temporal.vehicle.id.device)
  
        
            self.pretrained_model.eval()
            time_step_min=data_temporal.time_step[0]
            time_step_max=data_temporal.time_step[-1]
            lanelet_encoding_temporal_lst= []

            for t in range(time_step_max-time_step_min+1):
                data=data_temporal[t]
                occupancy_encodings=self.pretrained_model.encode(data)
                #data.vehicle_to_lanelet.edge_index
                #lanelet_encoding=[]
                lanelet_encoding=torch.empty(0,occupancy_encodings.shape[1],device=data_temporal.vehicle.id.device)

                
                #for i in data.vehicle_to_lanelet.edge_index[0,:]+data.vehicle.id[0]:
                counter=0
                for i in data.vehicle.id:    
                    
                    if i in ids_accepted:
                        #get the lanelet occupancy encoding between accepted vehicle and its current lanelet 
                        #lanelet_encoding.append(occupancy_encodings[data.vehicle_to_lanelet.edge_index[1,i-min_id]])

                        lanelet_encoding=torch.cat((lanelet_encoding,\
                        torch.unsqueeze((occupancy_encodings[data.vehicle_to_lanelet.edge_index[1,counter]]),dim=0)),dim=0)

                    counter = counter + 1
                lanelet_encoding_temporal_lst.append(lanelet_encoding)

            lanelet_encoding_temporal=torch.stack(lanelet_encoding_temporal_lst, dim=1)
                
            #lanelet_encoding_temporal=torch.Tensor(lanelet_encoding_temporal)
            out=torch.cat((x_t_accepted[:,:,:10],lanelet_encoding_temporal,x_t_accepted[:,:,10:]), dim=2)
            if out.shape[1]==1:
                out_x = out
                out_y = out
            else:
                out_x=out[:,:-1,:]
                out_y=out[:,1:,:]
            return out_x, out_y

def get_pos_temporal_sequence_for_vehicle(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:

    if data_temporal.vehicle is None:
        raise AttributeError("self.vehicle is None")
    x = data_temporal.vehicle.pos# type: ignore
    id = data_temporal.vehicle.id.squeeze() # type: ignore
    x_v: Tensor = x[id==vehicle_id, :]
    return x_v