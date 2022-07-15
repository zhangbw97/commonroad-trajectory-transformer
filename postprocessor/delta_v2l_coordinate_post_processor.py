from math import nan
import time
from typing import List, Optional, Type
from torch.tensor import Tensor
import torch
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor,BaseTemporalDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
import tutorials.train_trajectory_transformer.models.config as Config

def get_delta_lanelet_lateral_error(data_temporal:CommonRoadDataTemporal,ids_accepted:List[int]) -> Tensor:
    """_summary_

    Args:
        data_temporal (CommonRoadDataTemporal)
        vehicle_id (int)

    Returns:
        Tensor: the vehicle delta lateral error at each timestep relative to the initial state. 

    Note: 
        This function takes changing lanelet into account, such that the lateral error is accumulated 
        It works as long as the lanelet width is identical thoughout the lanelet, regardless the lanelet is linear or curve.
    """
    if data_temporal.vehicle is None:
        raise AttributeError("self.vehicle is None")  
    # vehicle_idx is the index of the element in vehicle.id.squeeze()
    # vehicle_id_first_idx is the first index of given vehicle_idx in v2l information
    vehicle_idx = []
    for id in ids_accepted:
        vehicle_idx.append((data_temporal[0].vehicle.id.squeeze() == id).nonzero(as_tuple=True)[0])
        # vehicle_id_first_idx.append((data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0])
    vehicle_idx = torch.stack(vehicle_idx).view(-1)
    initial_lanelet_lateral_error = data_temporal[0].v2l.v2l_lanelet_lateral_error.index_select(0,vehicle_idx)
    initial_lanelet_idx = data_temporal[0].v2l.edge_index[1,:].index_select(0,vehicle_idx)

    time_step_min=data_temporal.time_step[0]
    time_step_max=data_temporal.time_step[-1]
    x_v = torch.zeros(len(ids_accepted), time_step_max-time_step_min+1,device=data_temporal[0].v.id.device)
    for t in range(1,time_step_max-time_step_min+1):

        vehicle_idx = []
        for id in ids_accepted:
            vehicle_idx.append((data_temporal[t].vehicle.id.squeeze() == id).nonzero(as_tuple=True)[0])
        vehicle_idx = torch.stack(vehicle_idx).view(-1)
        current_lanelet_idx = data_temporal[t].v2l.edge_index[1,:].index_select(0,vehicle_idx)

        same_lanelet_v_mask = (current_lanelet_idx == initial_lanelet_idx).nonzero(as_tuple=True)[0]
        different_lanelet_v_mask = (current_lanelet_idx != initial_lanelet_idx).nonzero(as_tuple=True)[0]
        # current lanelet is the same as initial lanelet
        if len(same_lanelet_v_mask) != 0:
            x_v_same_lanelet = data_temporal[t].v2l.v2l_lanelet_lateral_error.index_select(0,vehicle_idx[same_lanelet_v_mask]) - initial_lanelet_lateral_error.index_select(0,same_lanelet_v_mask)
            # insert the value according to the mask
            x_v[:,t].masked_scatter_((current_lanelet_idx == initial_lanelet_idx),x_v_same_lanelet)
        # current lanelet is different from initial lanelet
        if len(different_lanelet_v_mask) != 0:
            lanelet_distance_vec = data_temporal[t].l.start_pos.index_select(0,current_lanelet_idx.index_select(0,different_lanelet_v_mask))\
                - data_temporal[t].l.start_pos.index_select(0,initial_lanelet_idx.index_select(0,different_lanelet_v_mask))
            lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation.index_select(0,initial_lanelet_idx.index_select(0,different_lanelet_v_mask))
            
            lanelet_lateral_distance = torch.zeros(len(different_lanelet_v_mask),1)
            for v in range(len(different_lanelet_v_mask)):    
                lanelet_lateral_distance[v,:] = torch.dot(lanelet_distance_vec[v], lanelet_lateral_orientation_vec[v]/torch.linalg.norm(lanelet_lateral_orientation_vec[v]))
            x_v_different_lanelet = lanelet_lateral_distance + data_temporal[t].v2l.v2l_lanelet_lateral_error.index_select(0,vehicle_idx[different_lanelet_v_mask]) - initial_lanelet_lateral_error.index_select(0,different_lanelet_v_mask) 
            # insert the value according to the mask
            
            x_v[:,t].masked_scatter_((current_lanelet_idx != initial_lanelet_idx),x_v_different_lanelet)

    return x_v

def get_delta_lanelet_arclength_abs(data_temporal:CommonRoadDataTemporal,ids_accepted:List[int]) -> Tensor:
    """_summary_

    Args:
        data_temporal (CommonRoadDataTemporal)
        vehicle_id (int)

    Returns:
        Tensor: the vehicle delta lanelet_arclength_abs at each timestep relative to the initial state. 
    
    Note: 
        This function takes changing lanelet into account, such that the longitudinal arclength is accumulated.
        This only works for linear lanelets, since the lanelet length is the difference of global start pos.
        TODO: lanelet length attribute can be used to solve the problem, but requiring to know the relationship between lanelets()
    """

    if data_temporal.vehicle is None:
        raise AttributeError("self.vehicle is None")  

    # vehicle_idx is the index of the element in vehicle.id.squeeze()
    # vehicle_id_first_idx is the first index of given vehicle_idx in v2l information
    vehicle_idx = []
    for id in ids_accepted:
        vehicle_idx.append((data_temporal[0].vehicle.id.squeeze() == id).nonzero(as_tuple=True)[0])
        # vehicle_id_first_idx.append((data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0])
    vehicle_idx = torch.stack(vehicle_idx).view(-1)
    #vehicle_idx = (data_temporal[0].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0] 
    #vehicle_id_first_idx=(data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
    initial_lanelet_arclength_abs = data_temporal[0].v2l.v2l_lanelet_arclength_abs.index_select(0,vehicle_idx)
    initial_lanelet_idx = data_temporal[0].v2l.edge_index[1,:].index_select(0,vehicle_idx)
    
 
    time_step_min=data_temporal.time_step[0]
    time_step_max=data_temporal.time_step[-1]

    x_v = torch.zeros(len(ids_accepted), time_step_max-time_step_min+1,device=data_temporal[0].v.id.device)
    for t in range(1,time_step_max-time_step_min+1):
        vehicle_idx = []
        for id in ids_accepted:
            vehicle_idx.append((data_temporal[t].vehicle.id.squeeze() == id).nonzero(as_tuple=True)[0])
            # vehicle_id_first_idx.append((data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0])
        vehicle_idx = torch.stack(vehicle_idx).view(-1)
        #vehicle_id_first_idx=(data_temporal[t].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
        current_lanelet_idx = data_temporal[t].v2l.edge_index[1,:].index_select(0,vehicle_idx)

        same_lanelet_v_mask = (current_lanelet_idx == initial_lanelet_idx).nonzero(as_tuple=True)[0]
        different_lanelet_v_mask = (current_lanelet_idx != initial_lanelet_idx).nonzero(as_tuple=True)[0]

        # current lanelet is the same as initial lanelet
        if len(same_lanelet_v_mask) != 0:
            x_v_same_lanelet = data_temporal[t].v2l.v2l_lanelet_arclength_abs.index_select(0,vehicle_idx[same_lanelet_v_mask]) - initial_lanelet_arclength_abs.index_select(0,same_lanelet_v_mask)
            
            x_v[:,t].masked_scatter_((current_lanelet_idx == initial_lanelet_idx),x_v_same_lanelet)
            
        # The resolution is pretty bad
        # if initial_lanelet.successor == current_lanelet_id:
        #     lanelet_longitudinal_distance = data_temporal[t].l.length[initial_lanelet_idx]
        # else:
        # condition 1: first line is initial lanelet idx
        # condition 2: second line is current lanelet idx
        # condition1 = data_temporal[t].l2l.edge_index[0] == initial_lanelet_idx
        # condition2 = data_temporal[t].l2l.edge_index[1] == current_lanelet_idx
        # #find the l2l edge_index, that both true or both false in condition 1 and 2 tt+ff and t 
        # joint_condition = torch.eq(condition1,condition2)
        # initial_current_edge_index = None
        # for i in joint_condition.nonzero():
        #     if condition1[i] == True:
        #         initial_current_edge_index = i 
        # # PREDECESSOR = 0, SUCCESSOR = 1
        # predecessor_mask = initial_current_edge_index is not None and data_temporal[t].l2l.lanelet_edge_type[initial_current_edge_index] == 0
       
            
        # lanelet_longitudinal_distance = data_temporal[0].l.length[initial_lanelet_idx]
        # x_v_successive_lanelet =  lanelet_longitudinal_distance.index_select(predecessor_mask)  + data_temporal[t].v2l.v2l_lanelet_arclength_abs.index_select(predecessor_mask) - initial_arclength_abs.index_select(predecessor_mask)
        
        # current lanelet is different from initial lanelet
        if len(different_lanelet_v_mask) != 0:
            lanelet_distance_vec = data_temporal[t].l.start_pos.index_select(0,current_lanelet_idx.index_select(0,different_lanelet_v_mask))\
                    - data_temporal[t].l.start_pos.index_select(0,initial_lanelet_idx.index_select(0,different_lanelet_v_mask))
            lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation.index_select(0,initial_lanelet_idx.index_select(0,different_lanelet_v_mask))
            lanelet_lateral_distance = torch.zeros(len(different_lanelet_v_mask),1)
            lanelet_longitudinal_distance = torch.zeros(len(different_lanelet_v_mask),1)
            for v in range(len(different_lanelet_v_mask)):    
                lanelet_lateral_distance[v,:] = torch.dot(lanelet_distance_vec[v], lanelet_lateral_orientation_vec[v]/torch.linalg.norm(lanelet_lateral_orientation_vec[v]))
                lanelet_longitudinal_distance[v,:] = torch.sqrt(lanelet_distance_vec[v,0] ** 2 + lanelet_distance_vec[v,1] ** 2 - lanelet_lateral_distance[v] ** 2)
            
            x_v_different_lanelet = lanelet_longitudinal_distance + data_temporal[t].v2l.v2l_lanelet_arclength_abs.index_select(0,vehicle_idx[different_lanelet_v_mask]) - initial_lanelet_arclength_abs.index_select(0,different_lanelet_v_mask) 
            # insert the value according to the mask
            x_v[:,t].masked_scatter_((current_lanelet_idx != initial_lanelet_idx),x_v_different_lanelet)
           
    return x_v

def get_delta_lanelet_lateral_error_per_v(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
    """_summary_

    Args:
        data_temporal (CommonRoadDataTemporal)
        vehicle_id (int)

    Returns:
        Tensor: the vehicle delta lateral error at each timestep relative to the initial state. 

    Note: 
        This function takes changing lanelet into account, such that the lateral error is accumulated 
        It works as long as the lanelet width is identical thoughout the lanelet, regardless the lanelet is linear or curve.
    """
    if data_temporal.vehicle is None:
        raise AttributeError("self.vehicle is None")  
    # vehicle_idx is the index of the element in vehicle.id.squeeze()
    # vehicle_id_first_idx is the first index of given vehicle_idx in v2l information
    vehicle_idx = (data_temporal[0].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0]
    vehicle_id_first_idx=(data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
    initial_lanelet_lateral_error = data_temporal[0].v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx]
    initial_lanelet_idx = data_temporal[0].v2l.edge_index[1,vehicle_id_first_idx]

    time_step_min=data_temporal.time_step[0]
    time_step_max=data_temporal.time_step[-1]
    x_v = torch.zeros(time_step_max-time_step_min+1,device=data_temporal[0].v.id.device)
    for t in range(1,time_step_max-time_step_min+1):
        vehicle_idx = (data_temporal[t].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0]
        vehicle_id_first_idx=(data_temporal[t].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]

        current_lanelet_idx = data_temporal[t].v2l.edge_index[1,vehicle_id_first_idx]

        if(current_lanelet_idx == initial_lanelet_idx):
            x_v[t] = data_temporal[t].v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx] - initial_lanelet_lateral_error
        else:
            lanelet_distance_vec = data_temporal[t].l.start_pos[current_lanelet_idx,:] - data_temporal[t].l.start_pos[initial_lanelet_idx,:]
            lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation[initial_lanelet_idx,:]
            
            lanelet_lateral_distance = torch.dot(lanelet_distance_vec, lanelet_lateral_orientation_vec/torch.linalg.norm(lanelet_lateral_orientation_vec))
            x_v[t] = lanelet_lateral_distance + data_temporal[t].v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx] - initial_lanelet_lateral_error


    return x_v

def get_delta_lanelet_arclength_abs_per_v(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
    """_summary_

    Args:
        data_temporal (CommonRoadDataTemporal)
        vehicle_id (int)

    Returns:
        Tensor: the vehicle delta lanelet_arclength_abs at each timestep relative to the initial state. 
    
    Note: 
        This function takes changing lanelet into account, such that the longitudinal arclength is accumulated.
        This only works for linear lanelets, since the lanelet length is the difference of global start pos.
        TODO: lanelet length attribute can be used to solve the problem, but requiring to know the relationship between lanelets()
    """

    if data_temporal.vehicle is None:
        raise AttributeError("self.vehicle is None")  

    # vehicle_idx is the index of the element in vehicle.id.squeeze()
    # vehicle_id_first_idx is the first index of given vehicle_idx in v2l information
    vehicle_idx = (data_temporal[0].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0] 
    vehicle_id_first_idx=(data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
    initial_arclength_abs = data_temporal[0].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx]
    initial_lanelet_idx = data_temporal[0].v2l.edge_index[1,vehicle_id_first_idx]
    

    time_step_min=data_temporal.time_step[0]
    time_step_max=data_temporal.time_step[-1]

    # x = data_temporal.v2l.v2l_lanelet_arclength_abs
    # ids = data_temporal.vehicle.id.squeeze()
    # x_v = x[ids==vehicle_id, :] - initial_arclength_abs
    x_v = torch.zeros(time_step_max-time_step_min+1,device=data_temporal[0].v.id.device)
    for t in range(1,time_step_max-time_step_min+1):
        vehicle_idx = (data_temporal[t].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0]
        vehicle_id_first_idx=(data_temporal[t].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
        current_lanelet_idx = data_temporal[t].v2l.edge_index[1,vehicle_id_first_idx]
        
        if(current_lanelet_idx == initial_lanelet_idx):
            x_v[t] = data_temporal[t].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx] - initial_arclength_abs
        else:
            # The resolution is pretty bad
            # if initial_lanelet.successor == current_lanelet_id:
            #     lanelet_longitudinal_distance = data_temporal[t].l.length[initial_lanelet_idx]
            # else:
            # condition 1: first line is initial lanelet idx
            # condition 2: second line is current lanelet idx
            condition1 = data_temporal[t].l2l.edge_index[0] == initial_lanelet_idx
            condition2 = data_temporal[t].l2l.edge_index[1] == current_lanelet_idx
            #find the l2l edge_index, that both true or both false in condition 1 and 2 tt+ff and t 
            joint_condition = torch.eq(condition1,condition2)
            initial_current_edge_index = None
            for i in joint_condition.nonzero():
                if condition1[i] == True:
                    initial_current_edge_index = i  
            if initial_current_edge_index is not None and data_temporal[t].l2l.lanelet_edge_type[initial_current_edge_index] == 0: 
                # PREDECESSOR = 0, SUCCESSOR = 1
                lanelet_longitudinal_distance = data_temporal[0].l.length[initial_lanelet_idx]
               
            else:
                lanelet_distance_vec = data_temporal[t].l.start_pos[current_lanelet_idx,:] - data_temporal[t].l.start_pos[initial_lanelet_idx,:]
                lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation[0,:]
                
                lanelet_lateral_distance = torch.dot(lanelet_distance_vec, lanelet_lateral_orientation_vec/torch.linalg.norm(lanelet_lateral_orientation_vec))
                lanelet_longitudinal_distance = torch.sqrt(lanelet_distance_vec[0] ** 2 + lanelet_distance_vec[1] ** 2 - lanelet_lateral_distance ** 2)

            x_v[t] = lanelet_longitudinal_distance + data_temporal[t].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx] - initial_arclength_abs
            if x_v[t] < 0:
                print("x_v < 0 ---t: ", t, "vehicle_id: ", vehicle_id)
                print("initial lanelet: ",initial_lanelet_idx ,"initial lanelet id: ",data_temporal[0].l.lanelet_id[initial_lanelet_idx] , "initial_arclength_abs: ", initial_arclength_abs)
                print("current lanelet: ",current_lanelet_idx , "current lanelet id: ",data_temporal[t].l.lanelet_id[current_lanelet_idx], "current_arclength_abs: ", data_temporal[t].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx])
                print("vehicle on multiple lanelet? vehicle on following lanelets " , data_temporal[t].v2l.edge_index[1,vehicle_id_first_idx])
                print("lanelet longtitudinal distance: ",lanelet_longitudinal_distance)
                print("")
            if x_v[t] == 0.0:
                print("x_v = 0 --- t: ", t, "vehicle_id: ", vehicle_id)
                print("initial lanelet: ",initial_lanelet_idx , "initial lanelet id: ",data_temporal[0].l.lanelet_id[initial_lanelet_idx] ,"initial_arclength_abs: ", initial_arclength_abs)
                print("current lanelet: ",current_lanelet_idx ,"current lanelet id: ",data_temporal[t].l.lanelet_id[current_lanelet_idx],"current_arclength_abs: ", data_temporal[t].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx])
                print("vehicle on multiple lanelet? vehicle on following lanelets " , data_temporal[t].v2l.edge_index[1,vehicle_id_first_idx])
                print("")
    return x_v

class DeltaV2LCoordinatePostProcessor(BaseTemporalDataPostprocessor):

    def __init__(
        self,
        training: bool = True
    ) -> None:

        super().__init__()
        self._training = training
    def __call__(
        self,
        samples: List[CommonRoadDataTemporal],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadDataTemporal]:
        # cuda0 = torch.device('cuda:0')
        # cpu = torch.device('cpu')
        #iterate_start_time = time.time()
        for data_temporal in samples:
            # data_temporal=data_temporal.to(cuda0)
            min_id=torch.min(data_temporal.vehicle.id)
            max_id=torch.max(data_temporal.vehicle.id)
            ids=list(range(min_id,max_id+1))

            ################ vehicle filter ###########################
            #accept vehicles that exist for 40 time steps
            ids_accepted=[]
            for i in ids:
                
                if self._training is False and data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0] == 1: 
                    ids_accepted.append(i)
                #In training or validation stage, only accept full sequence length
                elif data_temporal.get_node_features_temporal_sequence_for_vehicle(i).shape[0]==Config.SEQUENCE_LENGTH:
                    ids_accepted.append(i)
            data_temporal.v.accepted_id = ids_accepted
            #start_time=time.time()
        

            try:
                data_temporal.v2l.delta_lanelet_arclength_abs=get_delta_lanelet_arclength_abs(data_temporal,ids_accepted)
            except Exception:
                print("vectorized computation failed")
            else:
                data_temporal.v2l.delta_lanelet_arclength_abs=torch.cat([torch.unsqueeze(get_delta_lanelet_arclength_abs_per_v(data_temporal,i),0)  for i in ids_accepted], dim=0)
            #mid_time=time.time()
           
            
            try:
                data_temporal.v2l.delta_lanelet_lateral_error=get_delta_lanelet_lateral_error(data_temporal,ids_accepted)
            except Exception:
                print("vectorized computation failed")
            else:
                data_temporal.v2l.delta_lanelet_lateral_error=torch.cat([torch.unsqueeze(get_delta_lanelet_lateral_error_per_v(data_temporal,i),0)  for i in ids_accepted], dim=0)
            #end_time = time.time()
            #assert data_temporal.v2l.delta_lanelet_arclength_abs.min() == 0.0
        # iterate_end_time = time.time()
        # overal_iterate_time = iterate_end_time -iterate_start_time
        # instep_time_first = mid_time - start_time
        # instep_time_last = end_time - mid_time
        # print(f"overall iterate time is {overal_iterate_time}")
        # print(f"each iteratation time, first part: {instep_time_first}, second part: {instep_time_last}")

        return samples

