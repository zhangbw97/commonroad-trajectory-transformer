
from copy import copy
from math import nan
from random import choices
import string
import sys
import time
from typing import List, Optional, Type
import torch
import numpy as np
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseTemporalDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from sklearn.preprocessing import KBinsDiscretizer
import tutorials.train_trajectory_transformer.models.config as Config
EPS = 1e-5

class FeatureDistributionComputationPostProcessor(BaseTemporalDataPostprocessor):
    """
    Calculate min max information for training dataset and use them for normalization. 
    The normalization parameters for validation data and test data should be obtained in training data
    The min max information should also be used for unnormalization and discretization

    """
    def __init__(
        self,
        include_goal_distance_long_noego: bool = True,
        include_goal_distance_lat_noego: bool = True,
        include_delta_lanelet_arclength_abs: bool = True,
        include_delta_lanelet_lateral_error: bool = True,
        include_heading_error: bool = True,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_yaw_rate: bool = True,
        include_delta_arclength_abs: bool = True,
        include_delta_lateral_error: bool = True,
        n_bins: int = Config.N_BINS
    ) -> None:
        self._include_goal_distance_long_noego = include_goal_distance_long_noego
        self._include_goal_distance_lat_noego = include_goal_distance_lat_noego
        self._include_delta_lanelet_arclength_abs = include_delta_lanelet_arclength_abs
        self._include_delta_lanelet_lateral_error = include_delta_lanelet_lateral_error
        self._include_heading_error = include_heading_error
        self._include_velocity = include_velocity
        self._include_acceleration = include_acceleration
        self._include_yaw_rate = include_yaw_rate
        self._n_bins = n_bins
        self._include_delta_arclength_abs = include_delta_arclength_abs
        self._include_delta_lateral_error = include_delta_lateral_error

        super().__init__()


    def __call__(
        self,
        samples: List[CommonRoadDataTemporal],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadDataTemporal]:

        feature_distribution_dict = {}
        # delta_lanelet_arclengths_abs = torch.empty(0,device=samples[0].vehicle.id.device)
        # delta_lanelet_lateral_errors = torch.empty(0,device=samples[0].vehicle.id.device)
        # goal_distances_long_noego = torch.empty(0,device=samples[0].vehicle.id.device)
        # goal_distances_lat_noego = torch.empty(0,device=samples[0].vehicle.id.device)
        # delta_lanelet_arclengths_abs = torch.empty(0,device=samples[0].vehicle.id.device)
        # delta_lanelet_lateral_errors = torch.empty(0,device=samples[0].vehicle.id.device)
        # heading_errors = torch.empty(0,device=samples[0].vehicle.id.device)
        # velocities = torch.empty(0,device=samples[0].vehicle.id.device)
        # accelerations = torch.empty(0,device=samples[0].vehicle.id.device)
        # yaw_rates = torch.empty(0,device=samples[0].vehicle.id.device)

        #collect distribution for each feature
        start_time = time.time()
        #initialize
        MAX = 10000.0
        MIN = -10000.0
        feature_distribution_dict[("delta_lanelet_arclengths_abs","min")] = MAX
        feature_distribution_dict[("delta_lanelet_arclengths_abs","max")] = MIN
        feature_distribution_dict[("delta_lanelet_lateral_errors","min")] = MAX
        feature_distribution_dict[("delta_lanelet_lateral_errors","max")] = MIN
        feature_distribution_dict[("goal_distances_long_noego","min")] = MAX
        feature_distribution_dict[("goal_distances_long_noego","max")] = MIN
        feature_distribution_dict[("goal_distances_lat_noego","min")] = MAX
        feature_distribution_dict[("goal_distances_lat_noego","max")] = MIN
        feature_distribution_dict[("heading_error","min")] = MAX
        feature_distribution_dict[("heading_error","max")] = MIN
        feature_distribution_dict[("velocity","min")] = MAX
        feature_distribution_dict[("velocity","max")] = MIN 
        feature_distribution_dict[("acceleration","min")] = MAX
        feature_distribution_dict[("acceleration","max")] = MIN        
        feature_distribution_dict[("yaw_rate","min")] = MAX
        feature_distribution_dict[("yaw_rate","max")] = MIN    
        for data_temporal in samples:
            if self._include_delta_lanelet_arclength_abs:
                min = torch.min(data_temporal.v2l.delta_lanelet_arclength_abs)
                max = torch.max(data_temporal.v2l.delta_lanelet_arclength_abs)
                if min<feature_distribution_dict[("delta_lanelet_arclengths_abs","min")]:
                    feature_distribution_dict[("delta_lanelet_arclengths_abs","min")] = min
                if max>feature_distribution_dict[("delta_lanelet_arclengths_abs","max")]:
                    feature_distribution_dict[("delta_lanelet_arclengths_abs","max")] = max
                
            if self._include_delta_lanelet_lateral_error:
                min = torch.min(data_temporal.v2l.delta_lanelet_lateral_error)
                max = torch.max(data_temporal.v2l.delta_lanelet_lateral_error)
                if min<feature_distribution_dict[("delta_lanelet_lateral_errors","min")]:
                    feature_distribution_dict[("delta_lanelet_lateral_errors","min")] = min
                if max>feature_distribution_dict[("delta_lanelet_lateral_errors","max")]:
                    feature_distribution_dict[("delta_lanelet_lateral_errors","max")] = max
                
            for data in data_temporal:
                if self._include_goal_distance_long_noego:
                    min = torch.min(data.v.goal_distance_long_noego)
                    max = torch.max(data.v.goal_distance_long_noego)
                    if min<feature_distribution_dict[("goal_distances_long_noego","min")]:
                        feature_distribution_dict[("goal_distances_long_noego","min")] = min
                    if max>feature_distribution_dict[("goal_distances_long_noego","max")]:
                        feature_distribution_dict[("goal_distances_long_noego","max")] = max
                    
                if self._include_goal_distance_lat_noego:
                    min = torch.min(data.v.goal_distance_lat_noego)
                    max = torch.max(data.v.goal_distance_lat_noego)
                    if min<feature_distribution_dict[("goal_distances_lat_noego","min")]:
                        feature_distribution_dict[("goal_distances_lat_noego","min")] = min
                    if max>feature_distribution_dict[("goal_distances_lat_noego","max")]:
                        feature_distribution_dict[("goal_distances_lat_noego","max")] = max

                
                if self._include_heading_error:
                    min = torch.min(data.v.heading_error)
                    max = torch.max(data.v.heading_error)
                    if min<feature_distribution_dict[("heading_error","min")]:
                        feature_distribution_dict[("heading_error","min")] = min
                    if max>feature_distribution_dict[("heading_error","max")]:
                        feature_distribution_dict[("heading_error","max")] = max
                        
                if self._include_velocity:
                    min = torch.min(data.v.velocity)
                    max = torch.max(data.v.velocity)
                    if min<feature_distribution_dict[("velocity","min")]:
                        feature_distribution_dict[("velocity","min")] = min
                    if max>feature_distribution_dict[("velocity","max")]:
                        feature_distribution_dict[("velocity","max")] = max

                if self._include_acceleration:
                    min = torch.min(data.v.acceleration)
                    max = torch.max(data.v.acceleration)
                    if min<feature_distribution_dict[("acceleration","min")]:
                        feature_distribution_dict[("acceleration","min")] = min
                    if max>feature_distribution_dict[("acceleration","max")]:
                        feature_distribution_dict[("acceleration","max")] = max

                if self._include_yaw_rate:
                    min = torch.min(data.v.yaw_rate)
                    max = torch.max(data.v.yaw_rate)
                    if min<feature_distribution_dict[("yaw_rate","min")]:
                        feature_distribution_dict[("yaw_rate","min")] = min
                    if max>feature_distribution_dict[("yaw_rate","max")]:
                        feature_distribution_dict[("yaw_rate","max")] = max

        # for data_temporal in samples:
        #     if self._include_delta_lanelet_arclength_abs:
        #         delta_lanelet_arclengths_abs = torch.cat((delta_lanelet_arclengths_abs, torch.flatten(data_temporal.v2l.delta_lanelet_arclength_abs.view(1,-1))))
        #     if self._include_delta_lanelet_lateral_error:
        #         delta_lanelet_lateral_errors = torch.cat((delta_lanelet_lateral_errors, torch.flatten(data_temporal.v2l.delta_lanelet_lateral_error.view(1,-1))))
        #     for data in data_temporal:
        #         if self._include_goal_distance_long_noego:
        #             goal_distances_long_noego = torch.cat((goal_distances_long_noego,torch.flatten(data.v.goal_distance_long_noego)))
        #         if self._include_goal_distance_lat_noego:
        #             goal_distances_lat_noego = torch.cat((goal_distances_lat_noego,torch.flatten(data.v.goal_distance_lat_noego)))

                
        #         if self._include_heading_error:
        #             heading_errors = torch.cat((heading_errors,torch.flatten(data.v.heading_error)))
        #         if self._include_velocity:
        #             velocities = torch.cat((velocities,torch.flatten(data.v.velocity)))
        #         if self._include_acceleration:
        #             accelerations = torch.cat((accelerations,torch.flatten(data.v.acceleration)))
        #         if self._include_yaw_rate:
        #             yaw_rates = torch.cat((yaw_rates,torch.flatten(data.v.yaw_rate)))

        # feature_distribution_dict[("delta_lanelet_arclengths_abs","min")] = torch.min(delta_lanelet_arclengths_abs)
        # feature_distribution_dict[("delta_lanelet_arclengths_abs","max")] = torch.max(delta_lanelet_arclengths_abs)
        # feature_distribution_dict[("delta_lanelet_lateral_errors","min")] = torch.min(delta_lanelet_lateral_errors)
        # feature_distribution_dict[("delta_lanelet_lateral_errors","max")] = torch.max(delta_lanelet_lateral_errors)
        # feature_distribution_dict[("goal_distances_long_noego","min")] = torch.min(goal_distances_long_noego)
        # feature_distribution_dict[("goal_distances_long_noego","max")] = torch.max(goal_distances_long_noego)
        # feature_distribution_dict[("goal_distances_lat_noego","min")] = torch.min(goal_distances_lat_noego)
        # feature_distribution_dict[("goal_distances_lat_noego","max")] = torch.max(goal_distances_lat_noego)
        # feature_distribution_dict[("heading_error","min")] = torch.min(heading_errors)
        # feature_distribution_dict[("heading_error","max")] = torch.max(heading_errors)
        # feature_distribution_dict[("velocity","min")] = torch.min(velocities)
        # feature_distribution_dict[("velocity","max")] = torch.max(velocities)    
        # feature_distribution_dict[("acceleration","min")] = torch.min(accelerations)
        # feature_distribution_dict[("acceleration","max")] = torch.max(accelerations)        
        # feature_distribution_dict[("yaw_rate","min")] = torch.min(yaw_rates)
        # feature_distribution_dict[("yaw_rate","max")] = torch.max(yaw_rates)      
        end_time = time.time()
        print("computation time of min max value in dataset is : ", end_time-start_time)
        for key, value in feature_distribution_dict.items():
            if value == nan:
                raise ValueError(key, " has invalid number in dataset")
        return feature_distribution_dict

