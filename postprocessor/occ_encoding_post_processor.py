from typing import List, Optional, Type
import torch
from torch import Tensor
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseTemporalDataPostprocessor, BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from tutorials.train_geometric_model.models.occupancy.occupancy_model import OccupancyModel
from tutorials.train_trajectory_transformer.helper import get_delta_lanelet_arclength_abs, get_delta_lanelet_lateral_error
from sklearn.preprocessing import KBinsDiscretizer
EPS = 1e-5
class OCCEncodingPostProcessor(BaseTemporalDataPostprocessor):
    def __init__(
        self,
        max_distance: float = 100.0,
        max_sequence_length: int = 10,
        expand_route: bool = True,
        flatten: bool = True,
        model_filepath: str = 'tutorials/train_trajectory_transformer/pretrained_models/occ_model_highd.pt',
        include_path_decodings: bool = False,
        include_ego_vehicle_decodings: bool = False,
        decoding_resolution: int = 50,
        decoding_time_horizon: int = 60,
        occ_encoding_dim: int = 32
    ) -> None:
        self.max_distance = max_distance
        self.max_sequence_length = max_sequence_length
        self.flatten = flatten
        self.expand_route = expand_route
        self._model = OccupancyModel.load(model_filepath, device='cpu')
        self._include_decodings = include_path_decodings
        self._decoding_resolution = decoding_resolution
        self._decoding_time_horizon = decoding_time_horizon
        self._include_ego_vehicle_decodings = include_ego_vehicle_decodings
        self._occ_encoding_dim = occ_encoding_dim
        super().__init__()

    def get_routes(self, data:CommonRoadData, simulation):
        """_summary_

        Args:
            simulation (_type_): _description_
            data (CommonRoadData): _description_

        Returns:
            Tensors that represents:
            lanelet ids on current route, left route, right route, each with dimension [vehicle num x lanelet num each route]
            if a vehicle has no left or right route, -1 on corresponding position.
        """
        assert simulation is not None


        data_included_lanelet_ids_list = data.lanelet.lanelet_id.squeeze(1).tolist()
        data_included_lanelet_ids = set(data_included_lanelet_ids_list)
        data_included_lanelet_id_map = {lid: idx for idx, lid in enumerate(data_included_lanelet_ids_list)}

        num_vehicle = data.v.id.shape[0]
        current_lanelet_ids = torch.empty(0,1)

        lanelet_id_current_route = torch.zeros((num_vehicle,1),dtype=torch.int32)
        lanelet_id_left_route = torch.zeros((num_vehicle,1),dtype=torch.int32)
        lanelet_id_right_route = torch.zeros((num_vehicle,1),dtype=torch.int32)
        # initialize lanelet_id sequence in each route
        # if the vehicle don't have left or right adjacent lanelet, assign -1 to mark unexist left or right route
        for vehicle_idx in data.vehicle.indices:
            current_lanelet_idx = data.v2l.edge_index[1,((data.v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0])]
            if len(current_lanelet_idx)>1:
                current_lanelet_idx = current_lanelet_idx[0]
            try:
                current_lanelet_ids = torch.cat((current_lanelet_ids,data.l.lanelet_id[current_lanelet_idx].view(1,1)),dim=0)
            except KeyError as e:
                print("WARNING", repr(e))
                current_lanelet_ids = torch.cat((current_lanelet_ids,data.l.lanelet_id[current_lanelet_idx]),dim=0)
 
            lanelet_id_current_route[vehicle_idx,0] = data.l.lanelet_id[current_lanelet_idx].item()

            if simulation.lanelet_network.find_lanelet_by_id(int(current_lanelet_ids[-1])).adj_left_same_direction:
                lanelet_id_left = simulation.lanelet_network.find_lanelet_by_id(int(current_lanelet_ids[-1])).adj_left
                lanelet_id_left_route[vehicle_idx,0] = lanelet_id_left
            
            else: lanelet_id_left_route[vehicle_idx,0] = -1

            if simulation.lanelet_network.find_lanelet_by_id(int(current_lanelet_ids[-1])).adj_right_same_direction:
                lanelet_id_right = simulation.lanelet_network.find_lanelet_by_id(int(current_lanelet_ids[-1])).adj_right
                lanelet_id_right_route[vehicle_idx,0] = lanelet_id_right
            else: lanelet_id_right_route[vehicle_idx,0] = -1

        # expand routes to the end of scenario
        # expanding logic of left and right route is based on an assumption: if no successor of current lanelet, 
        # then also no successor on the left or right adjacent lanelet
        return_flag = False
        while True:
            append_current_route_successors = -1 * torch.ones((num_vehicle,1),dtype=torch.int32)
            append_left_route_successors = -1 * torch.ones((num_vehicle,1),dtype=torch.int32)
            append_right_route_successors = -1 * torch.ones((num_vehicle,1),dtype=torch.int32)
            for vehicle_idx in data.vehicle.indices:
                last_lanelet_current_route = simulation.lanelet_network.find_lanelet_by_id(lanelet_id_current_route[vehicle_idx,-1].item())
                current_successors = [s for s in last_lanelet_current_route.successor if s in data_included_lanelet_ids]
                if not current_successors:
                    return_flag = True
                    continue

                if lanelet_id_left_route[vehicle_idx,-1].item() != -1:
                    last_lanelet_left_route = simulation.lanelet_network.find_lanelet_by_id(lanelet_id_left_route[vehicle_idx,-1].item())
                    left_successors = [s for s in last_lanelet_left_route.successor if s in data_included_lanelet_ids]
                else: left_successors = [-1]

                if lanelet_id_right_route[vehicle_idx,-1].item() != -1:
                    last_lanelet_right_route = simulation.lanelet_network.find_lanelet_by_id(lanelet_id_right_route[vehicle_idx,-1].item())
                    right_successors = [s for s in last_lanelet_right_route.successor if s in data_included_lanelet_ids]
                else: right_successors = [-1]
                
                current_chosen = current_successors[0]
                append_current_route_successors[vehicle_idx] = current_chosen

                left_chosen = left_successors[0]
                append_left_route_successors[vehicle_idx] = left_chosen

                right_chosen = right_successors[0]
                append_right_route_successors[vehicle_idx] = right_chosen


            # [ vehicle_num x lanelet_num per route]
            lanelet_id_current_route = torch.cat((lanelet_id_current_route,append_current_route_successors),dim=1)
            lanelet_id_left_route = torch.cat((lanelet_id_left_route,append_left_route_successors),dim=1)
            lanelet_id_right_route = torch.cat((lanelet_id_right_route,append_right_route_successors),dim=1) 
            if return_flag:
                break
        return lanelet_id_current_route, lanelet_id_left_route, lanelet_id_right_route



    def setup_walks(self, data:CommonRoadData, simulation, lanelet_id_route: Tensor, vehicle_idx: int) -> CommonRoadData:
        """Generate walks attribute for each CommonroadData, pass to obtain occupancy embedding

        Args:
            data (CommonRoadData): _description_
            simulation (_type_): _description_
            lanelet_id_route (Tensor): _description_
            vehicle_idx (int): _description_

        Returns:
            CommonRoadData: _description_
        """
        lanelet_id_route= lanelet_id_route.tolist()
        assert len(lanelet_id_route) > 0
        assert lanelet_id_route[0] != -1

        data_included_lanelet_ids_list = data.lanelet.lanelet_id.squeeze(1).tolist()
        data_included_lanelet_ids = set(data_included_lanelet_ids_list)
        data_included_lanelet_id_map = {lid: idx for idx, lid in enumerate(data_included_lanelet_ids_list)}
        
        #     current_lanelet_candidates = [lid for lid in [data.l.lanelet_id[data.v2l.edge_index[1,(data.v2l.edge_index[0,:] == vehicle_idx).nonzero(as_tuple=True)[0]]].item()] if lid in lanelet_id_route]
        #     assert len(current_lanelet_candidates) > 0
        #     current_lanelet_id = current_lanelet_candidates[0]
        # except KeyError as e:
        #     print("WARNING", repr(e))
        current_lanelet_id = lanelet_id_route[0]
        current_lanelet_path = simulation.get_lanelet_center_polyline(current_lanelet_id)

        initial_arclength = current_lanelet_path.get_projected_arclength(data.v.pos[vehicle_idx])
        current_arclength = initial_arclength

        cumulative_arclength: float = 0.0
        route_buffer = []
        route_counter = lanelet_id_route.index(current_lanelet_id)

        while cumulative_arclength <= self.max_distance:
            remaining_lanelet_distance = current_lanelet_path.length - current_arclength
            distance_to_go = self.max_distance - cumulative_arclength
            current_lanelet_idx = data_included_lanelet_id_map[current_lanelet_id]

            if distance_to_go < remaining_lanelet_distance:
                delta = distance_to_go
                done = True
            else:
                delta = remaining_lanelet_distance
                done = False

            lanelet_signature = (
                current_lanelet_idx,
                current_lanelet_id,
                cumulative_arclength / self.max_distance,
                1 - current_arclength / current_lanelet_path.length,
                (current_arclength + delta) / current_lanelet_path.length,
                current_lanelet_path.length / self.max_distance
            )

            cumulative_arclength += delta
            route_buffer.append(lanelet_signature)

            if done:
                break
            elif route_counter == len(lanelet_id_route) - 1:
                break
            else:
                route_counter += 1
                current_lanelet_id = lanelet_id_route[route_counter]
                current_lanelet_path = simulation.get_lanelet_center_polyline(current_lanelet_id)
                current_arclength = 0.0

        has_lanelet_encodings = hasattr(data.lanelet, 'occupancy_encodings')

        walk_length = min(len(route_buffer), self.max_sequence_length)
        if len(route_buffer) == 0:
            print("error walk: ")
        if has_lanelet_encodings:
            encoding_dim = data.lanelet.occupancy_encodings.shape[1]
            encoding_sequence = torch.zeros((walk_length, encoding_dim), dtype=torch.float32, requires_grad=False)
        trajectory_sequence = torch.zeros((walk_length, 4), dtype=torch.float32, requires_grad=False)
        sequence_mask = torch.zeros((walk_length, 1), dtype=torch.bool, requires_grad=False)
        walks = torch.zeros((1, walk_length), dtype=torch.long, requires_grad=False)
        walk_start_length = torch.tensor(([initial_arclength]), dtype=torch.float32, requires_grad=False)

        data.walk_start_length = walk_start_length

        for i, (lanelet_idx, lanelet_id, cumulative_arclength, current_arclength, next_arclength, current_length) in enumerate(route_buffer):
            if i >= self.max_sequence_length:
                break
            if has_lanelet_encodings:
                encoding_sequence[i, :] = data.lanelet.occupancy_encodings[lanelet_idx, :]
            trajectory_sequence[i, :] = torch.tensor([
                cumulative_arclength, current_arclength, next_arclength, current_length
            ])
            sequence_mask[i, :] = True
            walks[:, i] = lanelet_idx

        if self.flatten:
            if has_lanelet_encodings:
                data.ego_encoding_sequence = encoding_sequence.flatten()
            data.ego_trajectory_sequence = trajectory_sequence.flatten()
            data.ego_trajectory_sequence_mask = sequence_mask.flatten()
            data.walks = walks.flatten()
        else:
            # if has_lanelet_encodings:
            #     data.ego_encoding_sequence = encoding_sequence
            data.ego_trajectory_sequence = trajectory_sequence
            data.ego_trajectory_sequence_mask = sequence_mask
            data.walks = walks

        return data



    def get_encoding(self, data:CommonRoadData, vehicle_idx: int, simulation):
        assert simulation is not None


        data.walk_velocity = torch.tensor(
            [data.v.velocity[vehicle_idx]],
            device=data.device,
            dtype=torch.float32
        )

        if self._model.config.path_conditioning:

            self._model.preprocess_conditioning(data, data.walks, data.walk_start_length)
            if len(data.walks)==0:
                print("setup walk failure")
            out = self._model.encode(data)
            z_ego_route, z_r, message_intensities = out
            # data_temporal.lanelet.occupancy_encodings = z_r
            # data_temporal.z_ego_route = z_ego_route.squeeze(0)
            # data_temporal.message_intensities = message_intensities

            if self._include_decodings:
                pass
                
            # occ_encoding for each vehicle(left route, current route, right route)
             
            if self._include_ego_vehicle_decodings:
                pass

        return z_ego_route

    def __call__(
        self,
        samples: List[CommonRoadDataTemporal],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadDataTemporal]:
        #data_temporal.lanelet.occupancy_encodings.shape[1]
        for data_temporal in samples:
            data = data_temporal[0]
            vehicle_num = data.v.id.shape[0]
        
            lanelet_id_current_route, lanelet_id_left_route, lanelet_id_right_route =self.get_routes(data, simulation)

            data_temporal.v.occ_encoding = torch.empty(vehicle_num,3,self._occ_encoding_dim + 1)
            flag_reallanelet = torch.zeros(1,1)
            flag_nolanelet = torch.ones(1,1)
            for vehicle_idx, vehicle_id in enumerate(data.v.id):
                route_length = sum(lid !=-1 for lid in lanelet_id_current_route[vehicle_idx])
                data = self.setup_walks(data, simulation, lanelet_id_current_route[vehicle_idx,:route_length], vehicle_idx)
                data_temporal.v.occ_encoding[vehicle_idx,0] = torch.cat((flag_reallanelet, self.get_encoding(data,vehicle_idx,simulation)),dim=1)
                if lanelet_id_left_route[vehicle_idx][0] == -1:
                    data_temporal.v.occ_encoding[vehicle_idx,1] = torch.cat((flag_nolanelet,torch.zeros(1,self._occ_encoding_dim)),dim=1)
                else:
                    data = self.setup_walks(data, simulation, lanelet_id_left_route[vehicle_idx,:route_length], vehicle_idx)
                    data_temporal.v.occ_encoding[vehicle_idx,1] =  torch.cat((flag_reallanelet, self.get_encoding(data,vehicle_idx,simulation)),dim=1)               
                if lanelet_id_right_route[vehicle_idx][0] == -1:
                    data_temporal.v.occ_encoding[vehicle_idx,2] = torch.cat((flag_nolanelet,torch.zeros(1,self._occ_encoding_dim)),dim=1)
                else:
                    data = self.setup_walks(data, simulation, lanelet_id_right_route[vehicle_idx,:route_length], vehicle_idx)
                    data_temporal.v.occ_encoding[vehicle_idx,2] = torch.cat((flag_reallanelet, self.get_encoding(data,vehicle_idx,simulation)),dim=1) 

        return samples

        