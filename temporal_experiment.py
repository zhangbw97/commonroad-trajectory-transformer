from __future__ import annotations
import inspect
import os
from pickle import NONE
import sys
import time

from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from tutorials.train_trajectory_transformer.postprocessor.feature_descritization_post_processor import FeatureDiscretizationPostProcessor
from tutorials.train_trajectory_transformer.postprocessor.feature_distribution_computation_post_processor import FeatureDistributionComputationPostProcessor

current_dirdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # type: ignore
framework_dir = os.path.dirname(os.path.dirname(current_dirdir))
sys.path.insert(0, framework_dir)

from typing import Any, Callable, Dict, Iterable, List, Optional, Type, cast

import shutil
from functools import partial
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import load_dill, save_dill
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment
from dataclasses import dataclass
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorsPipeline
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions
from tutorials.train_trajectory_transformer.models.config import SEQUENCE_LENGTH
EXPORT_FILENAME = 'experiment_config'
EXPORT_FILETYPE = 'pkl'
EXPORT_DATASET_INFO_FILENAME = 'dataset_info'
@dataclass
class TemporalGeometricExperimentConfig:
    traffic_extraction_options: TrafficExtractorOptions
    data_collector_cls: Type[ScenarioDatasetCollector]
    simulation_options: Optional[BaseSimulationOptions] = None
    preprocessors: Optional[T_ScenarioPreprocessorsPipeline] = None
    temporaldata_postprocessors: Optional[T_LikeBaseDataPostprocessor] = None # todo: type hint
    data_postprocessors: Optional[T_LikeBaseDataPostprocessor] = None # todo: type hint
    enable_anomaly_detection = False
    feature_distribution_dict: Optional[Dict] = None
class TemporalGeometricExperiment:
    def __init__(self, config: TemporalGeometricExperimentConfig) -> None:
        self._config = config
        
        self._extractor_factory=TemporalTrafficExtractorFactory(
            traffic_extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
                    # node_sorter=None
            )),
            options=TemporalTrafficExtractorOptions(collect_num_time_steps=SEQUENCE_LENGTH)
        )

        self._collector = config.data_collector_cls(
            extractor_factory=self._extractor_factory,
            scenario_preprocessors=config.preprocessors,
            simulation_options=self._config.simulation_options
        )
        self._feature_distribution_dict = None
    @property
    def config(self) -> TemporalGeometricExperimentConfig:
        return self._config
    
    def create_name(self) -> str:
        return f"{self._config.data_collector_cls.__name__}-{get_timestamp_filename()}"

    def pre_transform_scenario(
        self,
        scenario: Scenario, 
        planning_problem_set: PlanningProblemSet,
        max_samples: T_CountParam = Unlimited

    ) -> Iterable[CommonRoadDataTemporal]:
        """Extract traffic data and preprocess scenarios

        Args:
            scenario (Scenario): Scenario to be processed.
            planning_problem_set (PlanningProblemSet): Planning problem set.
            max_samples (int, optional): Max samples to be generated per scenario. Defaults to 1.

        Returns:
            Iterable[CommonRoadDataTemporal]

        """
        
        if self._config.temporaldata_postprocessors is not None or self._config.data_postprocessors is not None:

            samples = []
            
            for time_step, data in self._collector.collect(
                scenario,
                planning_problem_set=planning_problem_set,
                max_samples=max_samples
            ):
                simulation = self._collector._simulation
                if data is not None:
                    samples.append(data)

            if samples.count(None) == 0:
                if self._config.data_postprocessors is not None:
                            for data_post_processor in self._config.data_postprocessors:
                                start_time = time.time()
                                for sample in samples:
                                    sample = data_post_processor(sample,simulation)
                                end_time=time.time()
                                current_post_processor_name = str(type(data_post_processor))
                                print(f"calculating time of{current_post_processor_name} is :", end_time-start_time)
                if self._config.temporaldata_postprocessors is not None:
                    for temporaldata_post_processor in self._config.temporaldata_postprocessors:
                        start_time = time.time()
                        if isinstance(temporaldata_post_processor,FeatureDistributionComputationPostProcessor):
                            self._config.feature_distribution_dict = temporaldata_post_processor(samples)
                
                            if self._config.feature_distribution_dict is not None:
                                print("successfully add feature_distribution_dict to config")
                        elif isinstance(temporaldata_post_processor,FeatureDiscretizationPostProcessor):
                            samples = temporaldata_post_processor(samples,feature_distribution_dict=self._config.feature_distribution_dict)
                        else:
                            samples = temporaldata_post_processor(samples,simulation)
                           
                        end_time=time.time()
                        current_post_processor_name = str(type(temporaldata_post_processor))
                        print(f"calculating time of{current_post_processor_name} is :", end_time-start_time)
                yield from samples
        else:
            for time_step, data in self._collector.collect(
                    scenario=scenario,
                    planning_problem_set=planning_problem_set,
                    max_samples=max_samples
                ):
                if data is not None:
                    yield data


    def collect_traffic_dataset(
        self,
        scenario_dir: str,
        dataset_dir: str,
        overwrite: bool,
        pre_transform_workers: int,
        max_scenarios: Optional[T_CountParam] = Unlimited,
        cache_data: bool = False,
        max_samples_per_scenario: Optional[T_CountParam] = Unlimited,
    ) -> CommonRoadDataset[CommonRoadDataTemporal,CommonRoadDataTemporal]:
        if overwrite:
            shutil.rmtree(dataset_dir, ignore_errors=True)

        if max_samples_per_scenario is None:
            max_samples_per_scenario = Unlimited
        if max_scenarios is None:
            max_scenarios = Unlimited
        pre_transform_scenario = partial(self.pre_transform_scenario, max_samples=max_samples_per_scenario) if pre_transform_workers > 0 else None

        commonroad_dataset = CommonRoadDataset[CommonRoadDataTemporal, CommonRoadDataTemporal](
            raw_dir=scenario_dir,
            processed_dir=dataset_dir,
            pre_transform=pre_transform_scenario,
            pre_transform_progress=True,
            pre_transform_workers=pre_transform_workers,
            max_scenarios=max_scenarios,
            cache_data=cache_data
        )

        return commonroad_dataset

    @staticmethod
    def _get_file_path(directory: str, export_filename:str = EXPORT_FILENAME) -> str:
        return os.path.join(directory, export_filename + '.' + EXPORT_FILETYPE)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        experiment_path = self._get_file_path(directory)
        save_dill(self._config, experiment_path)
        return experiment_path

    def save_dataset_feature_distribution(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        dataset_info_path = self._get_file_path(directory,EXPORT_DATASET_INFO_FILENAME)
        save_dill(self._config.feature_distribution_dict, dataset_info_path)
        return dataset_info_path

    def load_dataset_feature_distribution(self,
                                         dataset_info_path: str, 
                                         experiment: TemporalGeometricExperiment
                                         ):
        dataset_info_path = self._get_file_path(dataset_info_path) if not dataset_info_path.endswith(EXPORT_FILETYPE) else dataset_info_path
        
        experiment.config.feature_distribution_dict = cast(Dict, load_dill(dataset_info_path)) if experiment.config.feature_distribution_dict is None else experiment.config.feature_distribution_dict
        return     

    @classmethod
    def load(cls, file_path: str, config: TemporalGeometricExperimentConfig = None) -> TemporalGeometricExperiment:
        file_path = cls._get_file_path(file_path) if not file_path.endswith(EXPORT_FILETYPE) else file_path
        config = cast(TemporalGeometricExperimentConfig, load_dill(file_path)) if config is None else config
        experiment = TemporalGeometricExperiment(config)
        return experiment 
