from sre_constants import SUCCESS
import warnings
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np
import torch
import math
from commonroad.geometry.shape import ShapeGroup, Shape
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import ScenarioID

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VehicleNodeFeatureParams, VehicleLaneletEdgeFeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.simulation.interfaces import BaseSimulation

EPS = 1e-1


class GoalAlignmentNoegoFeatureComputer(BaseFeatureComputer[VehicleNodeFeatureParams]):        
    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(
        self,
        include_goal_distance_longitudinal: bool = True,
        include_goal_distance_lateral: bool = True,
        logarithmic: bool = False
    ) -> None:
        if not any((
            include_goal_distance_longitudinal,
            include_goal_distance_lateral,
        )):
            raise ValueError("GoalAlignmentComputer doesn't include any features")

        self._include_goal_distance_longitudinal = include_goal_distance_longitudinal
        self._include_goal_distance_lateral = include_goal_distance_lateral


        self._logarithmic = logarithmic
        self._lanelet_network: Optional[LaneletNetwork] = None
        self._scenario_id: Optional[ScenarioID] = None
        self._undefined_features = self._return_undefined_features()
        
        super().__init__()

    def __call__(
        self,
        params: VehicleNodeFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        features: FeatureDict = {}
        lanelet_graph = simulation.lanelet_graph
        scenario = simulation.initial_scenario

        current_state_lanelet_id = scenario.lanelet_network.find_lanelet_by_position([params.state.position])[0][0]
        current_lanelet = simulation.find_lanelet_by_id(current_state_lanelet_id)
        final_state_lanelet_id = scenario.lanelet_network.find_lanelet_by_position([params.obstacle.prediction.trajectory.final_state.position])[0][0]
        final_lanelet = simulation.find_lanelet_by_id(final_state_lanelet_id)

        
        final_state_center_polyline = simulation.get_lanelet_center_polyline(final_state_lanelet_id)
        final_state_lanelet_arclength_abs = final_state_center_polyline.get_projected_arclength(params.obstacle.prediction.trajectory.final_state.position)
        final_state_lateral_error = final_state_center_polyline.get_lateral_distance(params.obstacle.prediction.trajectory.final_state.position)
        
        current_state_center_polyline = simulation.get_lanelet_center_polyline(current_state_lanelet_id)
        current_state_lanelet_arclength_abs = current_state_center_polyline.get_projected_arclength(params.state.position)
        current_state_lateral_error = current_state_center_polyline.get_lateral_distance(params.state.position)


        # compare the direction of vehicle lanechange movement and the lanelet right <-> left direction, if the vehicle change to the right lane
        # the lanelet distance should be positive 
        if np.dot((params.obstacle.prediction.trajectory.final_state.position-params.state.position),(final_lanelet.right_vertices[-1] - current_lanelet.left_vertices[-1])) > 0:
            sign = 1
        else: 
            sign = -1
        
        current_lanelet_orientation_vec = torch.tensor([torch.cos(torch.tensor(current_state_center_polyline.start_direction)),\
            torch.sin(torch.tensor(current_state_center_polyline.start_direction))])
        lanelet_startpoints_vec = torch.tensor(final_state_center_polyline.start - current_state_center_polyline.start)

        lanelet_longitudinal_difference = torch.dot( lanelet_startpoints_vec, current_lanelet_orientation_vec)
        lanelet_lateral_difference = torch.norm(torch.cross(torch.cat((lanelet_startpoints_vec,torch.tensor([0]))),\
            torch.cat((current_lanelet_orientation_vec,torch.tensor([0])))))
        # if len(final_lanelet.predecessor) == 0 or (len(current_lanelet.predecessor) != 0 and len(final_lanelet.predecessor) != 0) :
        #     # final lanelet is start lanelet of road  OR  both current lanelet and final lanelet is end lanelet of road
        #     # so the 2 lanelets are parallel 
        #     lanelet_longitudinal_difference=0
        #     lanelet_lateral_difference =  sign * np.linalg.norm(current_lanelet.left_vertices[-1] - final_lanelet.left_vertices[-1])

        # elif len(current_lanelet.predecessor) == 0 and len(final_lanelet.predecessor) != 0:
        #     # current lanelet is start lanelet of road and final lanelet is end lanelet of road
        #     lanelet_longitudinal_difference = current_state_center_polyline.length
        #     lanelet_lateral_difference =  sign * np.linalg.norm(current_lanelet.left_vertices[-1] - simulation.find_lanelet_by_id(final_lanelet.predecessor[0]).left_vertices[-1])
        
        arclength_to_goal = lanelet_longitudinal_difference.item() + final_state_lanelet_arclength_abs - current_state_lanelet_arclength_abs
        if arclength_to_goal<0:
            raise ValueError("goal should be in front of vehicle")
        lateral_error_to_goal = lanelet_lateral_difference.item() + final_state_lateral_error - current_state_lateral_error 
        

        if self._include_goal_distance_longitudinal:
            features[V_Feature.GoalDistanceLongitudinalNoego.value] = np.log(arclength_to_goal + EPS) if self._logarithmic else arclength_to_goal
        if self._include_goal_distance_lateral:
            features[V_Feature.GoalDistanceLateralNoego.value] =  lateral_error_to_goal

        return features

    def _return_undefined_features(self) -> FeatureDict:
        features: FeatureDict = {}

        if self._include_goal_distance_longitudinal:
            features[V_Feature.GoalDistanceLongitudinal.value] = 0.0
        if self._include_goal_distance_lateral:
            features[V_Feature.GoalDistanceLateral.value] = 0.0
        # if self._include_goal_distance:
        #     features[V_Feature.GoalDistance.value] = 0.0
        # if self._include_lane_changes_required:
        #     features[V_Feature.LaneChangesRequired.value] = 0.0
        # if self._include_lane_changes_required:
        #     features[V_Feature.LaneChangeDirectionRequired.value] = 0.0

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        scenario_id = simulation.current_scenario.scenario_id
        if self._scenario_id is None or scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._lanelet_network = simulation.current_scenario.lanelet_network
            self._shortest_paths = nx.shortest_path(simulation.lanelet_graph)
            self._lane_changes_required_cache: Dict[(int, int), (int, int)] = {}

