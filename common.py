from logging import Logger
from pathlib import Path
from torch_geometric.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union, TypeVar
from torch_geometric.data.data import BaseData

import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from commonroad_geometric.common.utils.seeding import set_global_seed
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputersContainer
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment
from commonroad_geometric.learning.geometric.training.geometric_trainer import GeometricTrainer
from commonroad_geometric.learning.geometric.training.training_utils import OptimizationMetricsCallback
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService, OptunaOptimizerService, SweepsOptimizerService
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService

T_Data = TypeVar("T_Data", bound=BaseData)
def optimizer_service(
    wandb_service: WandbService, 
    metrics: Union[str, List[str]]=['validation_average'], 
    directions: Union[str, List[str]]='minimize',
    n_trials=50,
    use_sweeps=False
) -> BaseOptimizerService:
    """Create an optimization service to simulate parameter optimization

    Args:
        wandb_service (WandbService): Weights and biases service
        metrics (Union[str, List[str]], optional): Metric to be optimized. Defaults to ['validation_average'].
        directions (Union[str, List[str]], optional): Enforce criteria for optimization service i.e. maximize or minimize the metric. Defaults to 'minimize'.
        n_trials (int, optional): Number of trials to be conducted. Defaults to 50.
        use_sweeps (bool, optional): Use wandb sweeps instead of optuna. Defaults to False.

    Returns:
        BaseOptimizerService: Returns optimization service
    """
    optimizer_metrics_callback = OptimizationMetricsCallback()
    if use_sweeps:
        optimization_service = SweepsOptimizerService(
            metric_callback=optimizer_metrics_callback,
            wandb_service=wandb_service
        )
    else:
        optimization_service = OptunaOptimizerService(
            directions=directions,
            wandb_service=wandb_service,
            metrics=metrics,
            metric_callback=optimizer_metrics_callback,
            n_trials=n_trials,
        )
    return optimization_service

class BaseCollectDatasetAndTrainModel():
    def __init__(self, 
        logger: Logger = None,
        max_epochs: int = 100,
        overfit: bool = False,
        max_optimize_samples = 1,
        dataset_split: Union[float, int] = 1,
        device: Union[str, torch.device] = None,
        validation_freq: int = 1,
        backward_freq: int = 1,
        validate_inner: int = 1,
        enable_multi_gpu: bool = False,
        verbose: int = 1
    ) -> None:
        """Provides an interface for initializing a model and training it with multi gpu and hyperparameter optimization support

        Args:
            model_dir (str): Directory where trained model is saved.
            gpu_count (int, optional): Number of gpus. Defaults to 1.
            logger (Logger, optional): Logger class. Defaults to None.
            max_epochs (int, optional): Maximum epochs per training cycle. Defaults to 100.
            overfit (bool, optional): Overfit the model by training over 1 sample for max_epochs. Defaults to False.
            max_optimize_samples (int, optional): Maximum samples to be used for optimization if applicable. Defaults to 1000.
            dataset_split (Union[float, int]): If float, the ratio between validation, train dataset samples. If integer, the number of validation samples. Defaults to 1.
        """
        self._gpu_count = torch.cuda.device_count()
        self._logger = logging.getLogger(__name__) if logger is None else logger
        self._max_epochs = max_epochs
        self._overfit = overfit
        self._max_optimize_samples = max_optimize_samples
        self._dataset_split = dataset_split
        self._device = device if device is not None else 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self._validation_freq = validation_freq
        self._backward_freq = backward_freq
        self._validate_inner = validate_inner
        self._enable_multi_gpu = enable_multi_gpu
        self._verbose = verbose

    def train(
        self,
        model_dir: str,
        experiment: GeometricExperiment,
        dataset: CommonRoadDataset, 
        model: BaseGeometric, 
        wandb_service: WandbService,
        model_kwargs: Dict[str, Any] = {}, 
        train_batch_size: int = 50, 
        optimizer_service=None,
        callbacks_computers: Optional[CallbackComputersContainer] = None,
        enable_rendering: bool = True,
        video_freq: int = 1000,
        video_length: int = 400,
        wandb_experiment_args: Dict = None,
        optimizer_state: Optional[Dict[str, Any]] = None
    ) -> Any:
        if self.multi_gpu():
            args = (model_dir, experiment, dataset, model, wandb_service, model_kwargs, train_batch_size, optimizer_service, optimizer_state, callbacks_computers, enable_rendering, video_freq, video_length, wandb_experiment_args)
            mp.spawn(self._train_gnn_model, args=args, nprocs=self._gpu_count, join=True)
        else:
            self._train_gnn_model(0, model_dir, experiment, dataset, model, wandb_service, model_kwargs, train_batch_size, optimizer_service, optimizer_state, callbacks_computers, enable_rendering, video_freq, video_length)

    def multi_gpu(self) -> bool:
        device = self._device if type(self._device) == str else self._device.type
        return self._gpu_count > 1 and device == 'cuda' and self._enable_multi_gpu

    def _train_gnn_model(self, rank, model_dir: str, experiment: GeometricExperiment, dataset: Dataset, model, wandb_service: WandbService, model_kwargs, train_batch_size, optimizer_service, optimizer_state, callbacks_computers, enable_rendering, video_freq, video_length, wandb_experiment_args=None):
        # TODO: design this better
        
        if self.multi_gpu():
            # Need to set master address and port for process spawning
            # see: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_batching.py
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=rank, world_size=self._gpu_count)
            wandb_service.start_experiment(**wandb_experiment_args)

        set_global_seed(0)

        if optimizer_service is not None:
            subset = dataset.index_select(torch.arange(len(dataset), dtype=torch.long, device="cpu")[:self._max_optimize_samples])
            dataset_train, dataset_validation = subset[:int(self._max_optimize_samples/2)], subset[int(self._max_optimize_samples/2):]
        else: 
            if self._overfit:
                dataset_validation, dataset_train = dataset.split(size=1)
                dataset_train = dataset_validation
            else:
                dataset_validation, dataset_train = dataset.split(size=self._dataset_split)

        train_sampler = DistributedSampler(dataset_train, num_replicas=self._gpu_count,
                                        rank=rank)
        validation_sampler = DistributedSampler(dataset_validation, num_replicas=self._gpu_count,
                                                rank=rank)

        train_loader = DataLoader(dataset_train, batch_size=train_batch_size,collate_fn=noop_collate_fn, sampler=train_sampler)
        validation_loader = DataLoader(dataset_validation, batch_size=train_batch_size,collate_fn=noop_collate_fn, sampler=validation_sampler)

        self._logger.info(f"Model: {model}")
        trainer = GeometricTrainer(
            experiment=experiment,
            model=model,
            output_dir=Path(model_dir),
            callbacks_computers=callbacks_computers,
            multi_gpu=self.multi_gpu(),
            validation_freq=self._validation_freq,
            backward_freq=self._backward_freq,
            validate_inner=self._validate_inner,
            enable_rendering=enable_rendering,
            video_freq=video_freq,
            video_length=video_length,
            verbose=self._verbose
        )
        
        results = trainer.train_orchestrator(
            train_loader,
            validation_loader,
            max_epochs=self._max_epochs,
            device=rank if self.multi_gpu() else self._device,
            optimizer_service=optimizer_service,
            optimizer_state=optimizer_state,
            **model_kwargs
        )

        wandb_service.log(vars(results))

        if optimizer_service is not None:
            optimizer_service.conclude_trial()

        if self._gpu_count > 1:        
            dist.destroy_process_group()

def noop_collate_fn(data_list: List[T_Data]) -> T_Data:
    assert len(data_list) == 1
    return data_list[0]