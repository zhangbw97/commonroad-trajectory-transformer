import inspect
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Type, Union
from unittest import mock
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions

from tutorials.train_geometric_model.common import BaseCollectDatasetAndTrainModel

current_dirdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # type: ignore
framework_dir = os.path.dirname(os.path.dirname(current_dirdir))
sys.path.insert(0, framework_dir)

import shutil
import torch
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.extraction.traffic.node_sorting.vehicle_node_sorter import VehicleNodeSorter, VehicleNodeSorterOptions
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILENAME
from commonroad_geometric.learning.training.wandb_service import WandbService
from commonroad_geometric.learning.training.git_features.git_feature_collector import GitFeatureCollector
from commonroad_geometric.learning.training.git_features.defaults import DEFAULT_GIT_FEATURE_COLLECTORS
from tutorials.train_geometric_model.models.dummy_model import DummyModel
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
# from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
# from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.edge_drawers import base_edge_drawer
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal

from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputersContainer, CallbackComputerContainerService
from commonroad_geometric.learning.geometric.training.callbacks.implementations.early_stopping_callback import EarlyStoppingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.epoch_checkpoint_callback import EpochCheckpointCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperimentConfig
from tutorials.train_trajectory_transformer.temporal_experiment import TemporalGeometricExperiment
import tutorials.train_trajectory_transformer.feature_config as Config
from tutorials.train_trajectory_transformer.config import POSTPROCESSORS
DATA_COLLECTOR_CLS = ScenarioDatasetCollector
SCENARIO_DIR = 'tutorials/test_scenarios/highd' if DATA_COLLECTOR_CLS is ScenarioDatasetCollector else 'tutorials/test_scenarios/highway_test'
DATASET_DIR = 'tutorials/train_trajectory_transformer/output/dataset_test_t100'




if __name__ == '__main__':
    shutil.rmtree(DATASET_DIR, ignore_errors=True)

    experiment_config = GeometricExperimentConfig(
        #traffic_extraction_options=TemporalTrafficExtractorOptions(collect_num_time_steps=COLLECT_NUM_TIME_STEPS),
        traffic_extraction_options=TrafficExtractorOptions(
                    vehicle_node_feature_computers=Config.V_FEATURE_COMPUTERS,
                    lanelet_node_feature_computers=Config.L_FEATURE_COMPUTERS,
                    vehicle_edge_feature_computers=Config.V2V_FEATURE_COMPUTERS,
                    lanelet_edge_feature_computers=Config.L2L_FEATURE_COMPUTERS,
                    vehicle_to_lanelet_edge_feature_computers=Config.V2L_FEATURE_COMPUTERS,
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50)),
        data_collector_cls=ScenarioDatasetCollector,
        preprocessors=None,
        postprocessors=None
    )
    experiment = TemporalGeometricExperiment(experiment_config)

    #collect CommonRoadDataset, which contains collected Iterable[CommonRoadDataTemporal]
    dataset = experiment.collect_traffic_dataset(
        scenario_dir=SCENARIO_DIR,
        dataset_dir=DATASET_DIR,
        overwrite=True,
        pre_transform_workers=4,
        max_scenarios=1,
        cache_data=True
    )
    print("Done exporting graph dataset")

    """
    dataset[0] is a CommonRoadDataTemporal instance from dataset, you can inspect its properties in DEBUG CONSOLE
    dataset[0][0] is a CommonRoadData instance, which corresponds to the first timestep in dataset[0]
    One example usage is to take each vehicle's trajectory in the graph and its node feature as a batch trajectories,
    the batch shape is [num_vehicle x time_length x feature_dim]
    """
    batch=dataset[0].get_node_features_temporal_sequence()
