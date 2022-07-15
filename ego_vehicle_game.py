import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.rendering import TrafficSceneRenderer
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.video_recording import save_images_from_frames, save_video_from_frames
from commonroad_geometric.rendering.types import T_Frame
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.steering_acceleration_control_space import SteeringAccelerationControlSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation, EgoVehicleSimulationFinishedException
from commonroad_geometric.simulation.ego_simulation.planning_problem import EgoRoute
from commonroad_geometric.simulation.ego_simulation.respawning.base_ego_vehicle_respawner import BaseEgoVehicleRespawner
from commonroad_geometric.simulation.ego_simulation.respawning.implementations.random_respawner import RandomRespawner
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_game.keyboard_input import UserQuitGameInterrupt, UserResetGameInterrupt, get_keyboard_action
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions


class EgoVehicleGame(AutoReprMixin):
    def __init__(
        self,
        scenario: Union[Scenario, str],
        simulation: Optional[BaseSimulation] = None,
        control_space: Optional[BaseControlSpace] = None,
        respawner: Optional[BaseEgoVehicleRespawner] = None,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        traffic_extractor_options: Optional[TrafficExtractorOptions] = None,
        traffic_renderer_options: Optional[TrafficSceneRendererOptions] = None,
        **sumo_simulation_kwargs: Any,
    ) -> None:
        control_space = control_space or SteeringAccelerationControlSpace()
        respawner = respawner or RandomRespawner()
        traffic_extractor_options = traffic_extractor_options or TrafficExtractorOptions(edge_drawer=VoronoiEdgeDrawer())
        if isinstance(scenario, str):
            scenario, planning_problem_set = CommonRoadFileReader(filename=scenario).open()

        self._renderer = TrafficSceneRenderer(
            options=traffic_renderer_options,
        )
        self._running: bool = False

        if simulation is None:
            simulation = SumoSimulation(
                initial_scenario=scenario,
                options=SumoSimulationOptions(
                    **sumo_simulation_kwargs,
                )
            )
        # simulation._options.on_step_renderer = self._renderer

        ego_route = EgoRoute(
            simulation=simulation,
            planning_problem_set=planning_problem_set
        )

        traffic_extractor = TrafficExtractor(
            simulation=simulation,
            options=traffic_extractor_options
        )

        self._ego_sim = EgoVehicleSimulation(
            simulation=simulation,
            ego_route=ego_route,
            control_space=control_space,
            respawner=respawner,
            traffic_extractor=traffic_extractor,
        )
        self._video_frames: List[T_Frame] = []

    @property
    def ego_simulation(self) -> EgoVehicleSimulation:
        return self._ego_sim

    @property
    def current_state(self) -> State:
        return self._ego_sim.ego_vehicle.state

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ego_vehicle(self) -> EgoVehicle:
        return self._ego_sim.ego_vehicle

    @property
    def ego_collision(self) -> bool:
        return self._ego_sim.check_if_has_collision()

    @property
    def ego_reached_goal(self) -> bool:
        return self._ego_sim.check_if_has_reached_goal()

    @property
    def renderer(self) -> TrafficSceneRenderer:
        return self._renderer

    def start(self) -> None:
        self._ego_sim.start()
        self._running = True

    def close(self) -> None:
        self._ego_sim.close()
        self._running = False

    def load_model():
        while 1:
            try:
                last_modified_ts = get_file_last_modified_datetime(model_path)
                if after_datetime is not None and last_modified_ts < after_datetime:
                    return None
                model = BaseGeometric.load(model_path, device='cpu')
                model.eval()
                break
            except Exception as e:
                time.sleep(0.1)
        return model, last_modified_ts

    # Loading model
    model, last_modified_ts = load_model(None)
    def step(self) -> None:
        # try:
        #     action = get_keyboard_action(renderer=self.renderer)
        # except UserResetGameInterrupt:
        #     self._ego_sim.reset()
        #     action = np.array([0.0, 0.0], dtype=np.float32)
        # except UserQuitGameInterrupt:
        #     self.close()
        #     return

        #TODO load model as render_model
        #  extract data by traffic extractor, taking goal position in planning problem into account
        #  beam search to obtain action
        
        try:
            self._ego_sim.step(action)
        except EgoVehicleSimulationFinishedException:
            self.close()
        render_kwargs: Dict[str, Any] = dict(
            ego_vehicle_vertices=self.ego_vehicle.vertices,
            overlays={
                "Scenario": self._ego_sim.current_scenario.scenario_id,
                "Timestep": self._ego_sim.current_time_step,
                'Action': action
            },
            to_rgb_array=True
        )
        self._ego_sim.extract_data()
        frame = self._ego_sim.render(
            renderer=self.renderer,
            render_params=RenderParams(
                render_kwargs=render_kwargs
            )
        )
        self._video_frames.append(frame)

    def save_video(self, output_file: str, save_pngs: bool = False) -> None:
        print(f"Saving video of last {len(self._video_frames)} frames to '{output_file}'")
        save_video_from_frames(frames=self._video_frames, output_file=output_file)
        if save_pngs:
            png_output_dir = os.path.join(os.path.dirname(output_file), 'pngs')
            os.makedirs(png_output_dir, exist_ok=True)
            save_images_from_frames(self._video_frames, output_dir=png_output_dir)
        self.clear_frames()

    def clear_frames(self) -> None:
        self._video_frames = []
