from math import nan
import torch
from torch.tensor import Tensor
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
import numpy as np
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal

from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation
from typing import Tuple,Union, overload
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
import tutorials.train_trajectory_transformer.models.config as Config
EPS = 1e-1

def revert_discretize( src, n_bins, min ,max):

    tgt = torch.empty_like(src)
    bin_width = (max - min) / n_bins
    if src.dim() ==1:
        for i in range(src.shape[0]):
            tgt[i] = min + src[i] * bin_width + bin_width/2
    if src.dim() == 2:
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                tgt[i,j] = min + src[i,j] * bin_width + bin_width/2
    return tgt


def get_delta_lanelet_lateral_error(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
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
            # compare the direction of vehicle lanechange movement and the lanelet right <-> left direction, if the vehicle change to the right lane
            # the lanelet distance should be positive 
            # if np.dot((data_temporal[t].v.pos[vehicle_idx]-data_temporal[0].v.pos[vehicle_idx]),\
            #     (data_temporal[0].l.right_vertices[initial_lanelet_idx][-1] - data_temporal[0].l.left_vertices[initial_lanelet_idx][-1])) > 0:

            # if torch.dot((data_temporal[t].v.pos[vehicle_idx]-data_temporal[0].v.pos[vehicle_idx]).squeeze(),data_temporal[0].l.lanelet_lateral_orientation[0,:]) > 0:
            #     sign = 1
            # else: 
            #     sign = -1
            # lanelet_distance = sign * torch.sqrt((data_temporal[t].l.start_pos[current_lanelet_idx,0] - data_temporal[t].l.start_pos[initial_lanelet_idx,0])**2+\
            #     (data_temporal[t].l.start_pos[current_lanelet_idx,1] - data_temporal[t].l.start_pos[initial_lanelet_idx,1])**2)
            lanelet_distance_vec = data_temporal[t].l.start_pos[current_lanelet_idx,:] - data_temporal[t].l.start_pos[initial_lanelet_idx,:]
            lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation[0,:]
            
            lanelet_lateral_distance = torch.dot(lanelet_distance_vec, lanelet_lateral_orientation_vec/torch.linalg.norm(lanelet_lateral_orientation_vec))
            x_v[t] = lanelet_lateral_distance + data_temporal[t].v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx] - initial_lanelet_lateral_error


    return x_v

def get_delta_lanelet_arclength_abs(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
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
            lanelet_distance_vec = data_temporal[t].l.start_pos[current_lanelet_idx,:] - data_temporal[t].l.start_pos[initial_lanelet_idx,:]
            lanelet_lateral_orientation_vec = data_temporal[0].l.lanelet_lateral_orientation[0,:]
            
            lanelet_lateral_distance = torch.dot(lanelet_distance_vec, lanelet_lateral_orientation_vec/torch.linalg.norm(lanelet_lateral_orientation_vec))
            lanelet_longitudinal_distance = torch.sqrt(lanelet_distance_vec[0] ** 2 + lanelet_distance_vec[1] ** 2 - lanelet_lateral_distance ** 2)

            x_v[t] = lanelet_longitudinal_distance + data_temporal[t].v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx] - initial_arclength_abs

        
    return x_v
def get_log_delta_lanelet_arclength_abs(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
    lanelet_arclength_abs = get_delta_lanelet_arclength_abs(data_temporal,vehicle_id)
    log_lanelet_arclength_abs = torch.log(lanelet_arclength_abs+EPS)
    if nan in log_lanelet_arclength_abs:
        raise ValueError("wrong log value")

    return log_lanelet_arclength_abs
def get_delta_lanelet_arclength_rel(data_temporal:CommonRoadDataTemporal, vehicle_id: int) -> Tensor:
    """_summary_

    Args:
        data_temporal (CommonRoadDataTemporal)
        vehicle_id (int)

    Returns:
        Tensor: the vehicle delta lanelet_arclength_rel at each timestep relative to the (onroute lanelet total length/initial lanelet length). 
    
    """
    lanelet_arclength_abs = get_delta_lanelet_arclength_abs(data_temporal,vehicle_id)
    vehicle_idx = (data_temporal[0].vehicle.id.squeeze() == vehicle_id).nonzero(as_tuple=True)[0]
    vehicle_id_first_idx=(data_temporal[0].v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
    initial_lanelet_id = data_temporal[0].v2l.edge_index[1,vehicle_id_first_idx]
   
    #lanelet_total_length = data_temporal[0].v.onroute_lanelet_total_length
    initial_lanelet_length = data_temporal[0].l.length[initial_lanelet_id].squeeze()
    return lanelet_arclength_abs / initial_lanelet_length
def get_pos_from_lanelet_coordinate(data: CommonRoadData, delta_lanelet_arclength_rel:Tensor, delta_lanelet_lateral_error:Tensor, simulation:ScenarioSimulation )-> Tensor:
    """_summary_

    Args:
        data (CommonRoadData): initial data to generate prediction
        lanelet_arclength_rel (Tensor): 
            lanelet_arclength_rel for all vehicles exist in initial CommonroadData, with all the prediction of N_SEQUENCE. dim = [B x T x 1]
        lanelet_lateral_error (Tensor): 
            lanelet_lateral_error for all vehicles exist in initial CommonroadData, with all the prediction of N_SEQUENCE. dim = [B x T x 1]
        simulation (ScenarioSimulation): 

    Returns:
        Tensor: predicted position in global frame for all vehicles. dim = [B x T x 2]

    """
    # initial_lanelet_arclength_rel = data.v.lanelet_arclength_rel
    # initial_lanelet_lateral_error = data.v.lanelet_lateral_error

    vehicle_ids=data.v.id.squeeze()
    lanelet_ids = data.l.lanelet_id
    lanelet_idx=data.v2l.edge_index[1,:]
    T = delta_lanelet_arclength_rel.shape[1]
    pos = torch.zeros(vehicle_ids.shape[0],T,2,device=data.v.id.device)
    for vehicle_idx in range(vehicle_ids.shape[0]):
        #select the first lanelet that the vehicle is on(perhaps any better way possible?)

        vehicle_id_first_idx=(data.v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
        initial_lanelet_arclength_rel = data.v2l.v2l_lanelet_arclength_rel[vehicle_id_first_idx]
        initial_lanelet_lateral_error = data.v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx]
        initial_lanelet_idx = lanelet_idx[vehicle_id_first_idx]
        # initial_lanelet_arclength_rel = data.v2l.v2l_lanelet_arclength_rel[vehicle_idx]
        # initial_lanelet_lateral_error = data.v2l.v2l_lanelet_lateral_error[vehicle_idx]
        # initial_lanelet_idx = lanelet_idx[vehicle_idx]
        pos_t = torch.zeros(T,2,device=data.v.id.device)
        for t in range(T):
            lanelet_arclength_rel = initial_lanelet_arclength_rel + delta_lanelet_arclength_rel[vehicle_idx,t]
            lanelet_lateral_error = initial_lanelet_lateral_error + delta_lanelet_lateral_error[vehicle_idx,t]
            if abs(lanelet_arclength_rel) > 1:

                # find l2l edges with the successor relationship 
                edge_filter = data.lanelet_to_lanelet.lanelet_edge_type == LaneletEdgeType.SUCCESSOR
                edge_index = data.lanelet_to_lanelet.edge_index[:, edge_filter.squeeze(1)]

                # find the successor lanelet of initial lanelet and get its center line
                current_lanelet_idx = edge_index[1,(edge_index[0,:] == initial_lanelet_idx.item()).nonzero(as_tuple=True)[0]]
                if current_lanelet_idx.nelement() != 0 :
                    lanelet_center_line=simulation.get_lanelet_center_polyline(lanelet_id=lanelet_ids[current_lanelet_idx].item()) 
                    waypoints=lanelet_center_line.waypoints
                    current_lanelet_arclength_rel = data.l.length[initial_lanelet_idx]*(lanelet_arclength_rel - 1) /lanelet_center_line.length
                    nearest_point=waypoints[round(abs(len(waypoints)*current_lanelet_arclength_rel).item())-1]
                    theta = lanelet_center_line.get_direction(arclength=current_lanelet_arclength_rel*lanelet_center_line.length)
                    pos_t[t,0] = nearest_point[0] + np.sin(theta) * lanelet_lateral_error
                    pos_t[t,1] = nearest_point[1] - np.cos(theta) * lanelet_lateral_error
                else:
                    # if the initial lanelet is the end lanelet of current scenario, the rendering for 
                    # the part outside initial lanelet can fall back to initial pos 
                    lanelet_center_line=simulation.get_lanelet_center_polyline(lanelet_id=lanelet_ids[initial_lanelet_idx].item()) 
                    waypoints=lanelet_center_line.waypoints
                    nearest_point=waypoints[round(abs(len(waypoints) * initial_lanelet_arclength_rel).item())-1]
                    theta = lanelet_center_line.get_direction(arclength = initial_lanelet_arclength_rel * lanelet_center_line.length)
                    pos_t[t,0] = nearest_point[0] + np.sin(theta) * initial_lanelet_lateral_error
                    pos_t[t,1] = nearest_point[1] - np.cos(theta) * initial_lanelet_lateral_error
            elif abs(lanelet_arclength_rel) <= 1:
                lanelet_center_line=simulation.get_lanelet_center_polyline(lanelet_id=lanelet_ids[initial_lanelet_idx].item()) 
                waypoints=lanelet_center_line.waypoints
                nearest_point=waypoints[round(abs(len(waypoints)*lanelet_arclength_rel).item())-1]
                theta = lanelet_center_line.get_direction(arclength=lanelet_arclength_rel*lanelet_center_line.length)
                pos_t[t,0] = nearest_point[0] + np.sin(theta) * lanelet_lateral_error
                pos_t[t,1] = nearest_point[1] - np.cos(theta) * lanelet_lateral_error
        pos[vehicle_idx,:,:] = torch.unsqueeze(pos_t, dim=0)     
    return pos

def get_pos_from_lanelet_coordinate(data: CommonRoadData,delta_lanelet_lateral_error:Tensor, simulation:ScenarioSimulation , delta_lanelet_arclength_abs = None,log_delta_lanelet_arclength_abs = None)-> Tensor:
    """_summary_

    Args:

        data (CommonRoadData): initial data to generate prediction
        lanelet_arclength_rel (Tensor): 
            lanelet_arclength_rel for all vehicles exist in initial CommonroadData, with all the prediction of N_SEQUENCE. dim = [B x T x 1]
        lanelet_lateral_error (Tensor): 
            lanelet_lateral_error for all vehicles exist in initial CommonroadData, with all the prediction of N_SEQUENCE. dim = [B x T x 1]
        simulation (ScenarioSimulation): 

    Returns:
        Tensor: predicted position in global frame for all vehicles. dim = [B x T x 2]

    """
    current_vehicle_ids = data.v.id.squeeze()
    lanelet_ids = data.l.lanelet_id
    lanelet_idx=data.v2l.edge_index[1,:]
    T = delta_lanelet_lateral_error.shape[1]
    pos = torch.zeros(current_vehicle_ids.shape[0],T,2,device=data.v.id.device)
    for vehicle_idx in range(current_vehicle_ids.shape[0]):
        #select the first lanelet that the vehicle is on(perhaps any better way possible?)
        vehicle_id_first_idx=(data.v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0][0]
        initial_lanelet_arclength_abs = data.v2l.v2l_lanelet_arclength_abs[vehicle_id_first_idx]
        initial_lanelet_lateral_error = data.v2l.v2l_lanelet_lateral_error[vehicle_id_first_idx]
        initial_lanelet_idx = lanelet_idx[vehicle_id_first_idx]
        initial_lanelet_length = data.l.length[initial_lanelet_idx]
        # initial_lanelet_arclength_rel = data.v2l.v2l_lanelet_arclength_rel[vehicle_idx]
        # initial_lanelet_lateral_error = data.v2l.v2l_lanelet_lateral_error[vehicle_idx]
        # initial_lanelet_idx = lanelet_idx[vehicle_idx]
        pos_t = torch.zeros(T,2,device=data.v.id.device)
        for t in range(T):
            #lanelet_arclength_abs = initial_lanelet_arclength_abs + np.exp(log_delta_lanelet_arclength_abs[vehicle_idx,t])-EPS
            if delta_lanelet_arclength_abs is not None:
                lanelet_arclength_abs = initial_lanelet_arclength_abs + delta_lanelet_arclength_abs[vehicle_idx,t]
            elif log_delta_lanelet_arclength_abs is not None:
                lanelet_arclength_abs = initial_lanelet_arclength_abs + torch.exp(log_delta_lanelet_arclength_abs[vehicle_idx,t]) - EPS
            lanelet_lateral_error = initial_lanelet_lateral_error + delta_lanelet_lateral_error[vehicle_idx,t]
            if lanelet_arclength_abs > initial_lanelet_length:
                find_current_lanelet = False
                route_lanelet_idx = [initial_lanelet_idx]
                
                # find l2l edges with the successor relationship 
                edge_filter = data.lanelet_to_lanelet.lanelet_edge_type == LaneletEdgeType.SUCCESSOR
                edge_index = data.lanelet_to_lanelet.edge_index[:, edge_filter.squeeze(1)]
                while(find_current_lanelet == False):
                    # find the successor lanelet of initial lanelet and get its center line
                    success_lanelet_idx = edge_index[1,(edge_index[0,:] == route_lanelet_idx[-1].item()).nonzero(as_tuple=True)[0]]
        
                    # nelement: num of element
                    if success_lanelet_idx.nelement() != 0 :
                        
                        lanelet_center_line=simulation.get_lanelet_center_polyline(lanelet_id=lanelet_ids[ success_lanelet_idx].item()) 
                        waypoints=lanelet_center_line.waypoints
                        cumulative_lanelet_length = 0
                        for ll in range(len(route_lanelet_idx)):
                            cumulative_lanelet_length = cumulative_lanelet_length + data.l.length[route_lanelet_idx[ll]]
                        current_lanelet_arclength_rel = (lanelet_arclength_abs - cumulative_lanelet_length)/lanelet_center_line.length
                        if current_lanelet_arclength_rel>1:
                            find_current_lanelet = False
                            route_lanelet_idx.append(success_lanelet_idx)
                            continue
                        else:
                            find_current_lanelet = True
                        #current_lanelet_arclength_rel = data.l.length[initial_lanelet_idx]*(lanelet_arclength_rel - 1) /lanelet_center_line.length
                        nearest_point=waypoints[round(abs(len(waypoints)*current_lanelet_arclength_rel).item())-1]
                        theta = lanelet_center_line.get_direction(arclength=current_lanelet_arclength_rel*lanelet_center_line.length)
                        pos_t[t,0] = nearest_point[0] + np.sin(theta) * lanelet_lateral_error
                        pos_t[t,1] = nearest_point[1] - np.cos(theta) * lanelet_lateral_error
                    else:
                        # if the initial lanelet is the end lanelet of current scenario, the rendering for 
                        find_current_lanelet = True
                        pos_t[t,0] = nan
                        pos_t[t,1] = nan
            else:
                lanelet_center_line=simulation.get_lanelet_center_polyline(lanelet_id=lanelet_ids[initial_lanelet_idx].item()) 
                waypoints=lanelet_center_line.waypoints
                lanelet_arclength_rel = lanelet_arclength_abs/initial_lanelet_length
                nearest_point=waypoints[round(abs(len(waypoints)*lanelet_arclength_rel).item())-1]
                theta = lanelet_center_line.get_direction(arclength=lanelet_arclength_rel*lanelet_center_line.length)
                pos_t[t,0] = nearest_point[0] + np.sin(theta) * lanelet_lateral_error
                pos_t[t,1] = nearest_point[1] - np.cos(theta) * lanelet_lateral_error
        pos[vehicle_idx,:,:] = torch.unsqueeze(pos_t, dim=0)     
    return pos
