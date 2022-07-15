
from copy import copy
from math import nan
from random import choices
import string
from typing import Dict, List, Optional, Type
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

def one_hot_discretize( src, n_bins, min ,max):

    bin_edges = np.linspace(min, max, n_bins + 1)
    tgt = torch.zeros(src.shape[0], n_bins)
    for i in range(src.shape[0]):
        tgt[i, np.searchsorted(bin_edges[1:-1], src[i], side="right")] = 1

    return tgt
def revert_one_hot_discretize( src, n_bins, min , max):

    tgt = torch.empty(src.shape[0])
    bin_edges = np.linspace(min, max, n_bins + 1)
    bin_width = (max - min) / n_bins
    for i in range(src.shape[0]):
        tgt[i] = min + np.dot(src[i,:] , bin_edges[:-1]) + bin_width/2
    return tgt

def discretize_and_normalize(src, n_bins, min ,max):

    bin_edges = np.linspace(min, max, n_bins + 1)
    for i in range(src.shape[0]):
        src[i] = np.searchsorted(bin_edges[1:-1], src[i], side="right")
    return src/n_bins

def discretize(src, n_bins, min ,max):
    
    bin_edges = np.linspace(min, max, n_bins + 1)
    if src.dim() == 1:
        for i in range(src.shape[0]):
            src[i] = np.searchsorted(bin_edges[1:-1], src[i], side="right")
    
    if src.dim() == 2:
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                src[i,j] = np.searchsorted(bin_edges[1:-1], src[i,j], side="right")
    return src

def revert_discretize( src, n_bins, min ,max):

    tgt = torch.empty_like(src)
    bin_width = (max - min) / n_bins
    if src.dim() ==1:
        for i in range(src.shape[0]):
            tgt[i] = min + src[i] * n_bins * bin_width + bin_width/2
    if src.dim() == 2:
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                tgt[i,j] = min + src[i,j] * n_bins * bin_width + bin_width/2
    return tgt

class FeatureDiscretizationPostProcessor(BaseTemporalDataPostprocessor):
    """
    Calculate min max information for training dataset and use them for normalization. 
    The normalization parameters for validation data and test data should be obtained in training data
    The min max information should also be used for unnormalization and discretization

    """
    def __init__(
        self,
        encoding_strategy: string ="Ordinal" ,#"One_hot"
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
        self._encoding_strategy = encoding_strategy
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
        ego_vehicle: Optional[EgoVehicle] = None,
        feature_distribution_dict: Dict = None
    ) -> List[CommonRoadDataTemporal]:

        assert feature_distribution_dict is not None
        # perform a uniform discretization transform of the dataset

        for data_temporal in samples:
            if self._include_delta_lanelet_arclength_abs:
                if self._encoding_strategy == "One_hot":
                    data_temporal.v2l.delta_lanelet_arclength_abs = one_hot_discretize(torch.squeeze(data_temporal.v2l.delta_lanelet_arclength_abs),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("delta_lanelet_arclengths_abs","min")],
                                    max = feature_distribution_dict[("delta_lanelet_arclengths_abs","max")] )
                elif self._encoding_strategy == "Ordinal":
                    data_temporal.v2l.delta_lanelet_arclength_abs = discretize(torch.squeeze(data_temporal.v2l.delta_lanelet_arclength_abs),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("delta_lanelet_arclengths_abs","min")],
                                    max = feature_distribution_dict[("delta_lanelet_arclengths_abs","max")] )    

            if self._include_delta_lanelet_lateral_error:
                if self._encoding_strategy == "One_hot":
                    data_temporal.v2l.delta_lanelet_lateral_error = one_hot_discretize(torch.squeeze(data_temporal.v2l.delta_lanelet_lateral_error),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("delta_lanelet_lateral_errors","min")],
                                    max = feature_distribution_dict[("delta_lanelet_lateral_errors","max")] )
                elif self._encoding_strategy == "Ordinal":
                    data_temporal.v2l.delta_lanelet_lateral_error = discretize(torch.squeeze(data_temporal.v2l.delta_lanelet_lateral_error),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("delta_lanelet_lateral_errors","min")],
                                    max = feature_distribution_dict[("delta_lanelet_lateral_errors","max")] )    

            for data in data_temporal:
                if self._include_goal_distance_long_noego:
                    if self._encoding_strategy == "One_hot":

                        data.v.x[:,25] = one_hot_discretize(torch.squeeze(data.v.goal_distance_long_noego),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("goal_distances_long_noego","min")],
                                    max = feature_distribution_dict[("goal_distances_long_noego","max")] )

                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,25] = discretize(torch.squeeze(data.v.goal_distance_long_noego),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("goal_distances_long_noego","min")],
                                    max = feature_distribution_dict[("goal_distances_long_noego","max")] )

                if self._include_goal_distance_lat_noego:
                    if self._encoding_strategy == "One_hot":
                        data.v.x[:,26] = one_hot_discretize(torch.squeeze( data.v.goal_distance_lat_noego),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("goal_distances_lat_noego","min")],
                                    max = feature_distribution_dict[("goal_distances_lat_noego","max")] )
                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,26] = discretize(torch.squeeze( data.v.goal_distance_lat_noego),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("goal_distances_lat_noego","min")],
                                    max = feature_distribution_dict[("goal_distances_lat_noego","max")] )       

                if self._include_heading_error:
                    if self._encoding_strategy == "One_hot":
                        data.v.x[:,11] = one_hot_discretize(torch.squeeze (data.v.heading_error),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("heading_error","min")],
                                    max = feature_distribution_dict[("heading_error","max")] )
                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,11] = discretize(torch.squeeze (data.v.heading_error),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("heading_error","min")],
                                    max = feature_distribution_dict[("heading_error","max")] )

                if self._include_velocity:
                    if self._encoding_strategy == "One_hot":
                        data.v.x[:,0] = one_hot_discretize(torch.squeeze (data.v.velocity),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("velocity","min")],
                                    max = feature_distribution_dict[("velocity","max")], )
                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,0] = discretize(torch.squeeze (data.v.velocity),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("velocity","min")],
                                    max = feature_distribution_dict[("velocity","max")] )   

                if self._include_acceleration:
                    if self._encoding_strategy == "One_hot":
                        data.v.x[:,1] = one_hot_discretize(torch.squeeze (data.v.acceleration),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("acceleration","min")],
                                    max = feature_distribution_dict[("acceleration","max")] )
                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,1] = discretize(torch.squeeze (data.v.acceleration),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("acceleration","min")],
                                    max = feature_distribution_dict[("acceleration","max")]
                                    )

                if self._include_yaw_rate:
                    if self._encoding_strategy == "One_hot":
                        data.v.x[:,3] = one_hot_discretize(torch.squeeze (data.v.yaw_rate),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("yaw_rate","min")],
                                    max = feature_distribution_dict[("yaw_rate","max")] )
                    elif self._encoding_strategy == "Ordinal":
                        data.v.x[:,3] = discretize(torch.squeeze (data.v.yaw_rate),
                                    n_bins=self._n_bins ,
                                    min = feature_distribution_dict[("yaw_rate","min")],
                                    max = feature_distribution_dict[("yaw_rate","max")] )

        return samples

