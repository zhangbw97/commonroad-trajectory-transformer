#Must come first/should be the first line
from __future__ import annotations

from typing import TYPE_CHECKING
from tutorials.train_trajectory_transformer.models.decoder.GPT2 import GPT2, GPT2LatentEmbedding, GPT2ProbDecoder

from tutorials.train_trajectory_transformer.models.encoder.mlp_occ_encoder import MLPOccEncoder
from tutorials.train_trajectory_transformer.models.encoder.occ_prediction_encoder import OccPredictionEncoder
from tutorials.train_trajectory_transformer.models.encoder.pretrained_encoder import PretrainedEncoder 

if TYPE_CHECKING:
    from tutorials.train_trajectory_transformer.models.trajectory_generator import TrajectoryGenerator
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tutorials.train_trajectory_transformer.models.config as Config
from commonroad_geometric.simulation.interfaces import ScenarioSimulation
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations.no_edge_drawer import NoEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from torch import Tensor
import tutorials.train_trajectory_transformer.models.config as Config

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:,:, [-1]]] = -float('Inf')
    return out
@torch.no_grad()
def sample_sequence(model: TrajectoryGenerator, data:CommonRoadDataTemporal, steps=39, temperature=1.0, topk = False , k=10) -> Tensor:
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """

    if isinstance(model.encoder, MLPOccEncoder):
        x , _, out_occupancy= model.encoder.forward(data)


    B,t,C = x.shape
    probs_cache = torch.empty((B,steps,C,Config.N_BINS),device=x.device)
    for t in range(steps):
        if isinstance(model.decoder, GPT2ProbDecoder):
            #[B x t x C x V]
            logits = model.decoder.forward(x, out_occupancy)  
            #take the logits for the single predicted time step in current forward loop
            #[B x C x V]
            logits = logits[:,-1]    
            if topk:
                logits = top_k_logits(logits , k)
            ## keep track of probabilities before modifying logits
            probs = logits.softmax(dim=-1)  
            # pluck the logits at the final step and scale by temperature

            ## sample from the distribution
            ## [ B x C ]
            indices = torch.empty(0, 1, probs.shape[1])
            for v in range(probs.shape[0]):
                indice = torch.multinomial(probs[v,:,:], num_samples=1)
                indices = torch.cat((indices,indice.view(1,1,-1)),dim=0)

            x = torch.cat((x,indices),dim=1)
            probs_cache[:,t,:] = probs

        #x_cond = x if x.get_node_features_temporal_sequence().shape[1] <= block_size else x[:, -block_size:] # crop context if needed
        # if isinstance(model.decoder, GPT2LatentEmbedding):
        #     pred = model.decoder.forward(x,occupancy_latent)
        # if isinstance(model.decoder, GPT2):
        #     pred = model.decoder.forward(x)

        # pred = torch.unsqueeze(pred[:, -1, :] / temperature, dim=1)

        # # append to the sequence and continue
        # x = torch.cat((x, pred), dim=1)
    
    return x

@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device)
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):

        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()