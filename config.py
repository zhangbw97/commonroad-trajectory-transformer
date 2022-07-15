from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations.no_edge_drawer import NoEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer, FullyConnectedEdgeDrawer, KNearestEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.feature_computers import DEFAULT_FEATURE_COMPUTERS
from commonroad_geometric.dataset.postprocessing.implementations.lanelet_occupancy_post_processor import LaneletOccupancyPostProcessor
from commonroad_geometric.simulation.base_simulation import Unlimited
from commonroad_geometric.dataset.preprocessing.implementations.lanelet_network_subset_preprocessor import LaneletNetworkSubsetPreprocessor
from commonroad_geometric.dataset.preprocessing.implementations.segment_lanelet_preprocessor import SegmentLaneletPreprocessor
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.acceleration_feature_computer import AccelerationFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.callables import ft_orientation, ft_vehicle_shape, ft_velocity
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.yaw_rate_feature_computer import YawRateFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.ego.goal_alignment_feature_computer import GoalAlignmentComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.vehicle_vertices_feature_computer import VehicleVerticesFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.vehicle_lanelet_connectivity_feature_computer import VehicleLaneletConnectivityComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.vehicle_lanelet_pose_feature_computer import VehicleLaneletPoseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.num_lanelet_assignments_feature_computer import NumLaneletAssignmentsFeatureComputer
from copy import copy

from tutorials.train_trajectory_transformer.models.feature_computers.goal_alignment_feature_computer import GoalAlignmentNoegoFeatureComputer
from tutorials.train_trajectory_transformer.postprocessor.feature_descritization_post_processor import FeatureDiscretizationPostProcessor
from tutorials.train_trajectory_transformer.postprocessor.delta_v2l_coordinate_post_processor import DeltaV2LCoordinatePostProcessor
from tutorials.train_trajectory_transformer.postprocessor.feature_distribution_computation_post_processor import FeatureDistributionComputationPostProcessor
from tutorials.train_trajectory_transformer.postprocessor.occ_encoding_post_processor import OCCEncodingPostProcessor

OUTPUT_DIR='tutorials/train_trajectory_transformer/output'
RENDER = False # lambda i : i < 100
DATA_COLLECTOR_CLS = ScenarioDatasetCollector
#SCENARIO_DIR = 'tutorials/test_scenarios/highd-test'
SCENARIO_DIR = 'tutorials/output/trajectories/USA101/scenarios'
EDGE_DRAWER = NoEdgeDrawer()#FullyConnectedEdgeDrawer(dist_threshold=30.0)
# DATASET_DIR = 'tutorials/train_trajectory_transformer/output/dataset_t100'
OVERWRITE= False
N_WORKERS=4
MAX_SCENARIOS = 1
MAX_EPOCHS = 10000
CACHE_DATA = True
LEARNING_RATE=6e-4
BATCH_SIZE=1
GRADIENT_CLIPPING_THRESHOLD = 0.2
BACKWARD_FREQ = 1
VALIDATION_SET_SIZE = 1
FEATURE_COMPUTERS = copy(DEFAULT_FEATURE_COMPUTERS)
#FEATURE_COMPUTERS.VehicleToLanelet = [VehicleLaneletPoseEdgeFeatureComputer()]
FEATURE_COMPUTERS.Vehicle = [
        ft_velocity,
        AccelerationFeatureComputer(),
        ft_orientation,
        YawRateFeatureComputer(),
        ft_vehicle_shape,
        VehicleLaneletPoseFeatureComputer(update_exact_interval=1),
        VehicleLaneletConnectivityComputer(),
        NumLaneletAssignmentsFeatureComputer(),
        VehicleVerticesFeatureComputer(),
        GoalAlignmentNoegoFeatureComputer()
    ]
LANELET_ASSIGNMENT_STRATEGY = LaneletAssignmentStrategy.ONLY_CENTER
PREPROCESSORS = None
# PREPROCESSORS = [
#     LaneletNetworkSubsetPreprocessor(radius=150.0),
#     SegmentLaneletPreprocessor()
# ]

from tutorials.train_trajectory_transformer.models.config import SEQUENCE_LENGTH

DISCRETIZATION_RESOLUTION = 50
TEMPORALDATA_POSTPROCESSORS=[
    OCCEncodingPostProcessor(),
    DeltaV2LCoordinatePostProcessor(),
    FeatureDistributionComputationPostProcessor(),
    FeatureDiscretizationPostProcessor(),
    
    ]
DATA_POSTPROCESSORS=[
        LaneletOccupancyPostProcessor(
        time_horizon=SEQUENCE_LENGTH,
        discretization_resolution=DISCRETIZATION_RESOLUTION
    ),
        #LaneletEgoSequencePostProcessor(max_distance=SEQUENCE_LENGTH, max_sequence_length=3, flatten=False),
        # OccupancyEncodingPostProcessor(
        #         model_filepath='tutorials/train_trajectory_transformer/pretrained_models/occ_model_highd.pt',
        #         #decoding_resolution=150 if args.hd_videos else 25,
        #         include_path_decodings=False,
        #         include_ego_vehicle_decodings=False,
        #         decoding_time_horizon=SEQUENCE_LENGTH
        #     )
    ]
