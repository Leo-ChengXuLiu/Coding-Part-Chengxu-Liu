"""Physics-object reconstruction and event-BDT feature packing."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch

from .sparse_multitask_gnn import ModelOutput


@dataclass(frozen=True)
class DecodeConfig:
    top_member_threshold: float = 0.5
    same_top_threshold: float = 0.5
    max_top_nodes: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.top_member_threshold <= 1.0:
            raise ValueError("top_member_threshold must be in [0, 1]")
        if not 0.0 <= self.same_top_threshold <= 1.0:
            raise ValueError("same_top_threshold must be in [0, 1]")
        if self.max_top_nodes < 1:
            raise ValueError("max_top_nodes must be positive")


@dataclass(frozen=True)
class DecodedEvent:
    top_node_indices: tuple[int, ...]
    charm_node_index: int | None
    bdt_features: dict[str, float]


def _p4_observables(p4: torch.Tensor, prefix: str) -> dict[str, float]:
    energy, px, py, pz = (float(value) for value in p4.detach().cpu())
    pt = (px * px + py * py) ** 0.5
    momentum = (pt * pt + pz * pz) ** 0.5
    mass = max(energy * energy - momentum * momentum, 0.0) ** 0.5
    eta = 0.5 * torch.log(
        torch.tensor((momentum + pz + 1.0e-12) / (momentum - pz + 1.0e-12))
    ).item()
    phi = torch.atan2(torch.tensor(py), torch.tensor(px)).item()
    return {
        f"{prefix}_energy": energy,
        f"{prefix}_pt": pt,
        f"{prefix}_eta": eta,
        f"{prefix}_phi": phi,
        f"{prefix}_mass": mass,
    }


def _wrapped_delta_phi(left: float, right: float) -> float:
    return (left - right + pi) % (2.0 * pi) - pi


def _top_group(
    top_probability: torch.Tensor,
    same_top_probability: torch.Tensor,
    edge_index: torch.Tensor,
    config: DecodeConfig,
) -> tuple[list[int], list[float]]:
    seed = int(top_probability.argmax())
    group = [seed]
    selected_edge_scores: list[float] = []
    left, right = edge_index.detach().cpu()
    edge_scores = same_top_probability.detach().cpu()
    top_scores = top_probability.detach().cpu()

    while len(group) < config.max_top_nodes:
        candidates: list[tuple[float, float, int]] = []
        group_set = set(group)
        for edge_id, (a_tensor, b_tensor) in enumerate(zip(left, right)):
            a, b = int(a_tensor), int(b_tensor)
            if (a in group_set) == (b in group_set):
                continue
            candidate = b if a in group_set else a
            edge_score = float(edge_scores[edge_id])
            node_score = float(top_scores[candidate])
            if (
                edge_score >= config.same_top_threshold
                and node_score >= config.top_member_threshold
            ):
                candidates.append((edge_score, node_score, candidate))
        if not candidates:
            break
        edge_score, _, candidate = max(candidates)
        group.append(candidate)
        selected_edge_scores.append(edge_score)
    return group, selected_edge_scores


def decode_event(
    output: ModelOutput,
    p4: torch.Tensor,
    edge_index: torch.Tensor,
    config: DecodeConfig = DecodeConfig(),
) -> DecodedEvent:
    """Reconstruct one top candidate, one recoil-c candidate, and flat BDT inputs.

    ``p4`` is ordered as ``[energy, px, py, pz]``. The function expects one
    event; batched outputs should be sliced event by event by the caller.
    """

    number_of_nodes = len(output.top_logits)
    if p4.shape != (number_of_nodes, 4):
        raise ValueError("p4 must have shape [number of nodes, 4]")
    if len(output.charm_logits) != number_of_nodes:
        raise ValueError("top and charm heads contain different node counts")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, number of edges]")
    if edge_index.shape[1] != len(output.same_top_logits):
        raise ValueError("edge index and same-top head contain different edge counts")
    if number_of_nodes == 0:
        raise ValueError("cannot decode an empty event")

    top_probability = output.top_logits.detach().sigmoid()
    charm_probability = output.charm_logits.detach().sigmoid()
    same_top_probability = output.same_top_logits.detach().sigmoid()
    top_nodes, selected_edges = _top_group(
        top_probability, same_top_probability, edge_index, config
    )
    top_set = set(top_nodes)
    remaining = [index for index in range(number_of_nodes) if index not in top_set]
    charm_node = (
        max(remaining, key=lambda index: float(charm_probability[index]))
        if remaining
        else None
    )

    top_p4 = p4[top_nodes].sum(dim=0)
    top_scores = top_probability[top_nodes]
    features = {
        "n_nodes": float(number_of_nodes),
        "n_edges": float(edge_index.shape[1]),
        "top_group_size": float(len(top_nodes)),
        "top_seed_score": float(top_probability[top_nodes[0]]),
        "top_score_mean": float(top_scores.mean()),
        "top_score_min": float(top_scores.min()),
        "top_edge_score_mean": (
            sum(selected_edges) / len(selected_edges) if selected_edges else 0.0
        ),
        "top_edge_score_min": min(selected_edges) if selected_edges else 0.0,
        **_p4_observables(top_p4, "top"),
    }

    if charm_node is None:
        for name in (
            "charm_score", "charm_energy", "charm_pt", "charm_eta", "charm_phi",
            "charm_mass", "tc_energy", "tc_pt", "tc_eta", "tc_phi", "tc_mass",
            "tc_delta_eta", "tc_delta_phi", "tc_delta_r", "tc_energy_asymmetry",
            "tc_pt_balance",
        ):
            features[name] = float("nan")
    else:
        charm_p4 = p4[charm_node]
        charm = _p4_observables(charm_p4, "charm")
        tc = _p4_observables(top_p4 + charm_p4, "tc")
        delta_eta = features["top_eta"] - charm["charm_eta"]
        delta_phi = _wrapped_delta_phi(features["top_phi"], charm["charm_phi"])
        energy_sum = features["top_energy"] + charm["charm_energy"]
        pt_sum = features["top_pt"] + charm["charm_pt"]
        features.update(
            {
                "charm_score": float(charm_probability[charm_node]),
                **charm,
                **tc,
                "tc_delta_eta": delta_eta,
                "tc_delta_phi": delta_phi,
                "tc_delta_r": (delta_eta * delta_eta + delta_phi * delta_phi) ** 0.5,
                "tc_energy_asymmetry": (
                    (features["top_energy"] - charm["charm_energy"]) / energy_sum
                    if energy_sum else 0.0
                ),
                "tc_pt_balance": (
                    abs(features["top_pt"] - charm["charm_pt"]) / pt_sum
                    if pt_sum else 0.0
                ),
            }
        )

    return DecodedEvent(tuple(top_nodes), charm_node, features)
