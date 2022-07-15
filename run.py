

import inspect
import os
import sys

current_dirdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # type: ignore
framework_dir = os.path.dirname(os.path.dirname(current_dirdir))
sys.path.insert(0, framework_dir)

from enum import Enum
from pathlib import Path
import argparse
import warnings
import torch
import shutil
from class_resolver import ClassResolver

from commonroad_geometric.common.class_extensions.nested_to_dict import nested_to_dict
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.debugging.warnings import debug_warnings
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILENAME
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputersContainer, CallbackComputerContainerService
from commonroad_geometric.learning.geometric.training.callbacks.implementations.debug_train_backward_gradients_callback import DebugTrainBackwardGradientsCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.early_stopping_callback import EarlyStoppingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.epoch_checkpoint_callback import EpochCheckpointCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.export_latest_model_callback import ExportLatestModelCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment, GeometricExperimentConfig
from commonroad_geometric.learning.geometric.training.training_utils import GradientClippingCallback
from commonroad_geometric.learning.training.git_features.defaults import DEFAULT_GIT_FEATURE_COLLECTORS
from commonroad_geometric.learning.training.git_features.git_feature_collector import GitFeatureCollector
from commonroad_geometric.learning.training.wandb_service import WandbService
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from tutorials.train_trajectory_transformer.common import BaseCollectDatasetAndTrainModel, optimizer_service
from tutorials.train_trajectory_transformer.temporal_experiment import TemporalGeometricExperiment,TemporalGeometricExperimentConfig
from tutorials.train_trajectory_transformer.models.trajectory_generator import TrajectoryGenerator
import tutorials.train_trajectory_transformer.config as Config

class RunMode(Enum):
    train = 'train'
    optimize = 'optimize'
    preprocess = 'preprocess'
    preprocess_train = 'preprocess-train'
    #Enjoy = 'enjoy'

    def __str__(self):
        return self.value

def main(args) -> None:

    print(f"Executing train_trajectory_transformer/run.py with m={args.m}, d={args.d}, scenario_dir={args.scenario_dir}")
    output_dir = os.path.join(args.output_dir, args.d)
    latest_global_dir = os.path.join(output_dir, 'model', "latest")
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, 'dataset_USA101_t40_goal_aligned_discrete_withocc')
    activation_resolver = ClassResolver(
        [TrajectoryGenerator],
        base=BaseGeometric,
        default=TrajectoryGenerator,
    )
    model_cls = activation_resolver.lookup(args.model)
    experiment_name = f"{model_cls.__name__}--{get_timestamp_filename()}"
    if args.warmstart:
        latest_model_path = os.path.join(latest_global_dir, MODEL_FILENAME)
        latest_optimizer_state_path = os.path.join(latest_global_dir, 'optimizer.pt')
        model = model_cls.load(latest_model_path)
        optimizer_state = torch.load(latest_optimizer_state_path)
    else:
        model = model_cls()
        optimizer_state = None
    model.train(True)

    if args.mode in {RunMode.preprocess, RunMode.preprocess_train}:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        print(f"Removed existing dataset from {dataset_dir}")

    experiment_config = TemporalGeometricExperimentConfig(
        traffic_extraction_options=TrafficExtractorOptions(
            edge_drawer=Config.EDGE_DRAWER,
            vehicle_node_feature_computers=Config.FEATURE_COMPUTERS.Vehicle,
            vehicle_edge_feature_computers=Config.FEATURE_COMPUTERS.VehicleToVehicle,
            lanelet_node_feature_computers=Config.FEATURE_COMPUTERS.Lanelet,
            lanelet_edge_feature_computers=Config.FEATURE_COMPUTERS.LaneletToLanelet,
            vehicle_to_lanelet_edge_feature_computers=Config.FEATURE_COMPUTERS.VehicleToLanelet
        ),
        data_collector_cls=ScenarioDatasetCollector,
        preprocessors=Config.PREPROCESSORS,
        temporaldata_postprocessors=Config.TEMPORALDATA_POSTPROCESSORS,
        data_postprocessors=Config.DATA_POSTPROCESSORS,
        simulation_options=ScenarioSimulationOptions(
            lanelet_assignment_order=Config.LANELET_ASSIGNMENT_STRATEGY
        )
    )
    experiment = TemporalGeometricExperiment(experiment_config)

    dataset = experiment.collect_traffic_dataset(
        scenario_dir=args.scenario_dir,
        dataset_dir=dataset_dir,
        overwrite=args.overwrite,
        pre_transform_workers=args.n_workers if args.mode in {RunMode.preprocess, RunMode.preprocess_train} else 0,
        max_scenarios=args.max_scenarios,
        cache_data=Config.CACHE_DATA,
        max_samples_per_scenario=args.max_samples_per_scenario
    )
    if args.mode == RunMode.preprocess:
        experiment.save_dataset_feature_distribution(str(dataset_dir))
        print()
        print("Done exporting graph dataset - now exiting. Rerun the script with mode 'train' to train the model on the collected dataset.")
        return
    if args.mode == RunMode.preprocess_train:
        experiment.save_dataset_feature_distribution(str(dataset_dir))
        print()
        print("Done exporting graph dataset - now continuing with training.")

    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base_trainer = BaseCollectDatasetAndTrainModel(
        max_epochs=args.epochs,
        device=device,
        #overfit=True,
        max_optimize_samples=len(dataset),
        validation_freq=args.validation_freq,
        backward_freq=args.backward_freq,
        validate_inner=args.validate_inner,
        dataset_split=args.dataset_split,
        enable_multi_gpu=args.multi_gpu,
        verbose=args.verbose   
    )

    project_name = f"train_trajectory_transformer_{model_cls.__name__.lower()}" 

    wandb_service = WandbService(disable=args.no_wandb, project_name=project_name)
    wandb_metadata = dict(
        training={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        experiment=nested_to_dict(experiment.config),
        model=nested_to_dict(model.config) if hasattr(model, 'config') else None
    )
    wandb_kwargs = {}

    if args.m is not None:
        experiment_name = experiment_name + '-' + args.m.replace(' ', '-')
    if base_trainer.multi_gpu():
        wandb_kwargs["group"] = f'{type(model).__name__}-DDP'

    wandb_experiment_args = {
        'name': experiment_name,
        'metadata': wandb_metadata,
        'include_timestamp': False,
        **wandb_kwargs
    }
    if not base_trainer.multi_gpu():
        experiment_name = wandb_service.start_experiment(
            **wandb_experiment_args
        )

    model_dir = os.path.join(output_dir, 'model', experiment_name)
    checkpoint_dir = os.path.join(model_dir , "checkpoints")
    latest_dir = os.path.join(model_dir , "latest")

    #feature_collector = GitFeatureCollector(DEFAULT_GIT_FEATURE_COLLECTORS)
    #features = feature_collector()
    from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, BaseCallbackParams
    class LogInfoCallback(BaseCallback[BaseCallbackParams]): 
        def __call__(
            self,
            params: BaseCallbackParams,
        ):
            from pprint import pprint
            pprint(params.info_dict)
            return

    callbacks_computers = CallbackComputersContainer(
        training_step_callbacks=CallbackComputerContainerService([
            ExportLatestModelCallback(directory=latest_dir, model_filename=MODEL_FILENAME, save_frequency=args.checkpoint_frequency),
            ExportLatestModelCallback(directory=latest_global_dir, model_filename=MODEL_FILENAME, save_frequency=args.checkpoint_frequency),
            LogWandbCallback(wandb_service=wandb_service),
            GradientClippingCallback(Config.GRADIENT_CLIPPING_THRESHOLD)
            # DebugTrainBackwardGradientsCallback(frequency=200)
        ]),
        validation_step_callbacks=CallbackComputerContainerService([
            LogInfoCallback()
        ]),
        logging_callbacks=CallbackComputerContainerService([LogWandbCallback(wandb_service=wandb_service)]),
        initialize_training_callbacks=CallbackComputerContainerService([WatchWandbCallback(wandb_service=wandb_service, log_freq=args.log_freq)]),
        checkpoint_callbacks=CallbackComputerContainerService([EpochCheckpointCallback(directory=checkpoint_dir, model_filename=MODEL_FILENAME)]),
        early_stopping_callbacks=CallbackComputerContainerService([EarlyStoppingCallback(after_epochs=2)]),
    )
    

    optimization_service = optimizer_service(
        wandb_service=wandb_service,
        use_sweeps=args.optimizer == 'wandb',
    ) if args.mode == RunMode.optimize else None

    base_trainer.train(
        model_dir=model_dir,
        experiment=experiment,
        dataset=dataset,
        model=model,
        wandb_service=wandb_service,
        train_batch_size=args.batch_size,
        callbacks_computers=callbacks_computers,
        optimizer_service=optimization_service,
        enable_rendering=not args.no_render,
        video_freq=args.video_freq,
        video_length=args.video_length,
        wandb_experiment_args=wandb_experiment_args,
        optimizer_state=optimizer_state
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train trajectory generator model.")
    parser.add_argument('mode', type=RunMode, choices=list(RunMode))
    parser.add_argument("--scenario-dir", type=Path, help="path to scenario directory used for training", default=Config.SCENARIO_DIR)
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR, type=Path, help="output directory for the experiment")
    parser.add_argument("--overwrite", action="store_true", help="remove and re-create the output directory before training")
    parser.add_argument("--warmstart", action="store_true", help="continue from latest checkpoint")
    parser.add_argument("--log-file", default="train-model.log", help="path to the log file")
    parser.add_argument("--seed", type=int, help="integer for seeding the random number generator")
    parser.add_argument("--n-workers", type=int, help="number of processes", default=1)
    parser.add_argument('--model',
        type=str,
        default='TrajectoryGenerator',
        const='TrajectoryGenerator',
        nargs='?',
        choices=['TrajectoryGenerator'],
        help='model to train'
    )
    parser.add_argument('--optimizer',
        type=str,
        default='wandb',
        const='wandb',
        nargs='?',
        choices=['optuna', 'wandb'],
        help='which optimizer to use for hyperparameter tuning'
    )
    parser.add_argument("-m", type=str, help="experiment message (will be appended to experiment name)", required=True)
    parser.add_argument("-d", type=str, help="dataset suffix allowing storing and training on separate datasets", default='USA101-t40')
    parser.add_argument("--device", type=str, default='auto', help="torch device")
    parser.add_argument("--profile", action="store_true", help="profiles code")
    parser.add_argument("--debug", action="store_true", help="activates debug logging")
    parser.add_argument("--no-wandb", action="store_true", help="disable metric tracking")
    parser.add_argument("--no-warn", action="store_true", help="disable warnings")
    parser.add_argument('--checkpoint-frequency', type=int, help="how often to save model", default=10)
    parser.add_argument('--epochs', type=int, help="number of training epochs", default=Config.MAX_EPOCHS)
    parser.add_argument('--batch-size', type=int, help='minibatch size for training', default=Config.BATCH_SIZE)
    parser.add_argument('--video-freq', type=int, help="how often to record video", default=1000)
    parser.add_argument('--video-length', type=int, help="how long videos to record", default=400)
    parser.add_argument('--max-samples-per-scenario', type=int, help="max samples per scenario")
    parser.add_argument('--multi-gpu',  action="store_true", help="enable training on multiple gpus")
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--max-scenarios', type=int, help="max scenarios")
    parser.add_argument('--log-freq', type=int, help="wandb watch logging frequency", default=100)
    parser.add_argument("--no-render", action="store_true", help="disable rendering")
    parser.add_argument("--validation-freq", type=int, help="frequency with which the model should be evaluated against the validation set", default=1)
    parser.add_argument("--backward-freq", type=int, help="frequency with which the model will do gradient step", default=Config.BACKWARD_FREQ)
    parser.add_argument("--dataset-split", type=int, help="size of validation set", default=Config.VALIDATION_SET_SIZE)
    parser.add_argument("--validate-inner", action="store_true", help="whether to evaluate the model on the validation set within the training loop")
    args = parser.parse_args()


    def run_main() -> None:
        if args.profile:
            from commonroad_geometric.debugging.profiling import profile
            profile(main, dict(args=args))
        else:
            main(args)

    if args.no_warn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_main()
    elif args.debug:
        debug_warnings(run_main)
    else:
        run_main()