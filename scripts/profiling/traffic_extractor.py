import inspect
import os
import sys

current_dirdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # type: ignore
framework_dir = os.path.dirname(os.path.dirname(current_dirdir))
sys.path.insert(0, framework_dir)

import timeit
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.simulation.interfaces import SumoSimulation
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulationOptions
from commonroad_geometric.debugging.profiling import profile

INPUT_SCENARIO = 'tutorials/test_scenarios/osm_crawled/DEU_Munich_1-100.xml'
PRESIMULATION_STEPS = 50


def setup_extractor() -> TrafficExtractor:
    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=SumoSimulationOptions(
            presimulation_steps=PRESIMULATION_STEPS
        )
    )
    simulation.start()

    traffic_extractor = TrafficExtractor(
        simulation=simulation,
        options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer(dist_threshold=75.0),
        )
    )

    return traffic_extractor


def extract_data(traffic_extractor: TrafficExtractor) -> CommonRoadData:
    return traffic_extractor.extract(TrafficExtractionParams(
        index=0,
    ))



if __name__ == '__main__':

    extractor = setup_extractor()
    def target() -> CommonRoadData:
        return extract_data(extractor)

    def main() -> None:
        timer = timeit.Timer(target)  
        print(f"timeit: {timer.timeit(50):.2f}s")

    profile(main)
