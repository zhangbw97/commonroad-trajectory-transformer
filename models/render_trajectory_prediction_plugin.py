from math import nan
import torch
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D
import numpy as np
from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.rendering.defaults import DEFAULT_OBSTACLE_COLOR
from commonroad_geometric.rendering.types import RenderParams, T_ColorTuple
import tutorials.train_trajectory_transformer.models.config as Config


class RenderTrajectoryPredictionPlugin(BaseRendererPlugin):
    def __init__(
        self,
        initialstate_color: T_ColorTuple = (1.0, 0.4, 0.4, 1.0),
        radius: float = 0.8,
        initial_radius: float = 1.0,
        hist_line_color: float = (1.0, 1.0, 1.0),
        pred_line_color: float = (1.0, 0.4, 0.4),
        line_width = 0.3,
        line_rendering = True
    ) -> None:

        self._initialstate_color = initialstate_color,
        self._radius = radius
        self._initial_radius = initial_radius
        self._hist_line_color = hist_line_color
        self._pred_line_color = pred_line_color
        self._line_width = line_width
        self._line_rendering = line_rendering
        #self._rng_cache = CachedRNG(np.random.random)
    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
 
        pred_pos = params.render_kwargs.get('pred_pos')
        current_pos = pred_pos[:,0,:]
        
        for id in range(pred_pos.shape[0]):
            # if self._rng_cache == None:
            #     self._rng_cache = CachedRNG(np.random.random)
            # self._circle_color= self._rng_cache(key=id,n=1)


            viewer.draw_circle(
                    origin=current_pos[id,:],
                    radius=self._initial_radius,
                    filled = False,
                    color=(1.0, 0.4, 0.4, 1.0),#self._initialstate_color,
                    outline=False,
                    linecolor=self._hist_line_color,
                    linewidth=None
                )
        #[B,T,2]
        if self._line_rendering is False:
            for id in range(pred_pos.shape[0]):
  
                for t in range(pred_pos.shape[1]):
                    if pred_pos[id,t,0] == nan or pred_pos[id,t,1] == nan:
                            continue
                    viewer.draw_circle(
                            origin=pred_pos[id,t,:],
                            radius=self._radius,
                            color=self._initialstate_color,
                            outline=False,
                            linecolor=self._pred_line_color,
                            linewidth=None
                        )

        else:
            for id in range(pred_pos.shape[0]):

                pred_waypoints_lst = []              
                for pred_t in range(pred_pos.shape[1]):
                    pred_pos_x=pred_pos[id,pred_t,0]
                    pred_pos_y=pred_pos[id,pred_t,1]
                    if pred_pos_x == nan or pred_pos_y == nan:
                        continue
                    pred_waypoints_lst.append((pred_pos_x,pred_pos_y))

                viewer.draw_polyline(
                pred_waypoints_lst,
                linewidth=self._line_width,
                color=self._pred_line_color
                )