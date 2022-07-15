from typing import List, Optional, Type
import torch
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseTemporalDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from tutorials.train_trajectory_transformer.helper import get_delta_lanelet_arclength_abs, get_delta_lanelet_lateral_error
from sklearn.preprocessing import KBinsDiscretizer
EPS = 1e-5
class FeatureNormalizationPostProcessor(BaseTemporalDataPostprocessor):
    """
    Calculate min max information for training dataset and use them for normalization. 
    The normalization parameters for validation data and test data should be obtained in training data
    The min max information should also be used for unnormalization
    For descritization, also use min max to do uniform descritization
    """
    def __init__(
        self,
        normalize_goal_distance_long_noego: bool = True,
        normalize_goal_distance_lat_noego: bool = True,
        normalize_delta_lanelet_arclength_abs: bool = True,
        normalize_delta_lanelet_lateral_error: bool = True,
        normalize_heading_error: bool = True,
        normalize_velocity: bool = True,
        normalize_acceleration: bool = True,
        normalize_yaw_rate: bool = True,
    ) -> None:
        self._normalize_goal_distance_long_noego = normalize_goal_distance_long_noego
        self._normalize_goal_distance_lat_noego = normalize_goal_distance_lat_noego
        self._normalize_delta_lanelet_arclength_abs = normalize_delta_lanelet_arclength_abs
        self._normalize_delta_lanelet_lateral_error = normalize_delta_lanelet_lateral_error
        self._normalize_heading_error = normalize_heading_error
        self._normalize_velocity = normalize_velocity
        self._normalize_acceleration = normalize_acceleration
        self._normalize_yaw_rate = normalize_yaw_rate

        
        super().__init__()
    def normalize(self, src, min ,max):
        return (src - min)/(max - min)

    def unnormalize(self, src, min ,max):
        return src*(max - min) + min

    def compute_distribution(self, attr):
        ...
    def __call__(
        self,
        samples: List[CommonRoadDataTemporal],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadDataTemporal]:

        feature_distribution_dict = {}
        goal_distances_long_noego = torch.empty(0,device=samples[0].vehicle.id.device)
        goal_distances_lat_noego = torch.empty(0,device=samples[0].vehicle.id.device)
        delta_lanelet_arclengths_abs = torch.empty(0,device=samples[0].vehicle.id.device)
        delta_lanelet_lateral_errors = torch.empty(0,device=samples[0].vehicle.id.device)
        heading_errors = torch.empty(0,device=samples[0].vehicle.id.device)
        velocities = torch.empty(0,device=samples[0].vehicle.id.device)
        accelerations = torch.empty(0,device=samples[0].vehicle.id.device)
        yaw_rates = torch.empty(0,device=samples[0].vehicle.id.device)

        #collect distribution for each feature
        for data_temporal in samples:
            # if self._normalize_delta_lanelet_arclength_abs:
            #     get_delta_lanelet_arclength_abs(data_temporal    
            #     delta_lanelet_arclengths_abs = torch.cat((delta_lanelet_arclengths_abs,delta_lanelet_arclength_abs))
            for data in data_temporal:
                if self._normalize_goal_distance_long_noego:
                    goal_distances_long_noego = torch.cat((goal_distances_long_noego,torch.flatten(data.v.goal_distance_long_noego)))
                if self._normalize_goal_distance_lat_noego:
                    goal_distances_lat_noego = torch.cat((goal_distances_lat_noego,torch.flatten(data.v.goal_distance_lat_noego)))

                
                if self._normalize_heading_error:
                    heading_errors = torch.cat((heading_errors,torch.flatten(data.v.heading_error)))
                if self._normalize_velocity:
                    velocities = torch.cat((velocities,torch.flatten(data.v.velocity)))
                if self._normalize_acceleration:
                    accelerations = torch.cat((accelerations,torch.flatten(data.v.acceleration)))
                if self._normalize_yaw_rate:
                    yaw_rates = torch.cat((yaw_rates,torch.flatten(data.v.yaw_rate)))

        feature_distribution_dict[("goal_distances_long_noego","min")] = torch.min(goal_distances_long_noego)
        feature_distribution_dict[("goal_distances_long_noego","max")] = torch.max(goal_distances_long_noego)
        feature_distribution_dict[("goal_distances_lat_noego","min")] = torch.min(goal_distances_lat_noego)
        feature_distribution_dict[("goal_distances_lat_noego","max")] = torch.max(goal_distances_lat_noego)
        feature_distribution_dict[("heading_error","min")] = torch.min(heading_errors)
        feature_distribution_dict[("heading_error","max")] = torch.max(heading_errors)
        feature_distribution_dict[("velocity","min")] = torch.min(velocities)
        feature_distribution_dict[("velocity","max")] = torch.max(velocities)    
        feature_distribution_dict[("acceleration","min")] = torch.min(accelerations)
        feature_distribution_dict[("acceleration","max")] = torch.max(accelerations)        
        feature_distribution_dict[("yaw_rate","min")] = torch.min(yaw_rates)
        feature_distribution_dict[("yaw_rate","max")] = torch.max(yaw_rates)       
        # apply normalization
       
        for data_temporal in samples:
            for data in data_temporal:
                if self._normalize_goal_distance_long_noego:
                    data.v.x[:,25] = self.normalize(src= data.v.goal_distance_long_noego,
                     min = feature_distribution_dict[("goal_distances_long_noego","min")],
                     max = feature_distribution_dict[("goal_distances_long_noego","max")] )
                    
                if self._normalize_goal_distance_lat_noego:
                    data.v.x[:,25] = self.normalize(src= data.v.goal_distance_long_noego,
                     min = feature_distribution_dict[("goal_distances_long_noego","min")],
                     max = feature_distribution_dict[("goal_distances_long_noego","max")] )
                    
                if self._normalize_heading_error:
                    data.v.x[:,11] = self.normalize(src= data.v.heading_error,
                     min = feature_distribution_dict[("heading_error","min")],
                     max = feature_distribution_dict[("heading_error","max")] )

                if self._normalize_velocity:
                    data.v.x[:,0] = self.normalize(src= data.v.velocity,
                     min = feature_distribution_dict[("velocity","min")],
                     max = feature_distribution_dict[("velocity","max")] )
                if self._normalize_acceleration:
                    data.v.x[:,1] = self.normalize(src= data.v.acceleration,
                     min = feature_distribution_dict[("acceleration","min")],
                     max = feature_distribution_dict[("acceleration","max")] )
                if self._normalize_yaw_rate:
                    data.v.x[:,3] = self.normalize(src= data.v.yaw_rate,
                     min = feature_distribution_dict[("yaw_rate","min")],
                     max = feature_distribution_dict[("yaw_rate","max")] )
        return samples, feature_distribution_dict

