#!/usr/bin/env python3
"""
XGBoost POC for Protein Interaction Prediction

A proof-of-concept script that uses XGBoost for temporal link prediction,
accepting the same preprocessed input files as RE-Net and producing 
compatible output for the existing analysis pipeline.

Features used for prediction (all use ONLY t-1 or earlier data):
- Edge temporal state (binary contact history with extended lags)
- Edge persistence / recency metrics
- Multi-scale temporal stability (duty cycles, flip counts at W5, W10, W20)
- Edge-level historical statistics
- Trend/momentum features
- Node context (degrees, degree changes, historical averages)
- Graph structure (Jaccard, Adamic-Adar, preferential attachment, etc.)
- Network effects (node activity rates, neighbor edge cascades, clustering, triangles)

Evaluation mode:
- Autoregressive/holdout forecasting: test predictions use only train+valid 
  history plus prior predicted contacts (not ground truth test data)
"""

import argparse
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Quadruple:
    """A single interaction record: (subject, relation, object, time)"""
    subject: int
    relation: int
    obj: int
    time: int


# =============================================================================
# Data Loading
# =============================================================================

def load_quadruples(file_path: str) -> List[Quadruple]:
    """Load quadruples from a data file."""
    quadruples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                quadruples.append(Quadruple(
                    subject=int(parts[0]),
                    relation=int(parts[1]),
                    obj=int(parts[2]),
                    time=int(parts[3])
                ))
    return quadruples


def load_stat(file_path: str) -> Tuple[int, int]:
    """Load statistics from stat.txt."""
    with open(file_path, 'r') as f:
        parts = f.read().strip().split()
        return int(parts[0]), int(parts[1])


def get_all_timestamps(quadruples_list: List[List[Quadruple]]) -> List[int]:
    """Get sorted list of all unique timestamps."""
    all_times = set()
    for quads in quadruples_list:
        for q in quads:
            all_times.add(q.time)
    return sorted(all_times)


def get_candidate_edges(quadruples_list: List[List[Quadruple]]) -> Set[Tuple[int, int]]:
    """Collect all unique (subject, object) pairs that appear anywhere."""
    edges = set()
    for quads in quadruples_list:
        for q in quads:
            edge = (min(q.subject, q.obj), max(q.subject, q.obj))
            edges.add(edge)
    return edges


def build_contact_matrix(
    quadruples: List[Quadruple],
    candidate_edges: Set[Tuple[int, int]],
    all_timestamps: List[int]
) -> Dict[Tuple[int, int], Dict[int, int]]:
    """Build contact matrix: contact[edge][time] = 0 or 1"""
    contact = {edge: {t: 0 for t in all_timestamps} for edge in candidate_edges}
    for q in quadruples:
        edge = (min(q.subject, q.obj), max(q.subject, q.obj))
        if edge in contact and q.time in contact[edge]:
            contact[edge][q.time] = 1
    return contact


def build_adjacency_at_time(
    quadruples: List[Quadruple],
    timestamps: List[int]
) -> Dict[int, Dict[int, Set[int]]]:
    """Build adjacency list for each timestamp."""
    adj = {t: defaultdict(set) for t in timestamps}
    for q in quadruples:
        adj[q.time][q.subject].add(q.obj)
        adj[q.time][q.obj].add(q.subject)
    return adj


def build_node_to_edges(candidate_edges: Set[Tuple[int, int]]) -> Dict[int, Set[Tuple[int, int]]]:
    """Build mapping from nodes to their incident candidate edges."""
    node_to_edges = defaultdict(set)
    for edge in candidate_edges:
        node_to_edges[edge[0]].add(edge)
        node_to_edges[edge[1]].add(edge)
    return dict(node_to_edges)


def update_adjacency_with_predictions(
    adj: Dict[int, Dict[int, Set[int]]],
    predictions_at_time: List[Tuple[int, int]],
    time: int
) -> None:
    """Update adjacency structure with predicted edges at a given time."""
    if time not in adj:
        adj[time] = defaultdict(set)
    for node_i, node_j in predictions_at_time:
        adj[time][node_i].add(node_j)
        adj[time][node_j].add(node_i)


# =============================================================================
# Historical Statistics (computed once from training data)
# =============================================================================

def compute_edge_historical_stats(
    contact: Dict[Tuple[int, int], Dict[int, int]],
    train_valid_timestamps: List[int],
    node_to_edges: Dict[int, Set[Tuple[int, int]]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute historical statistics for each edge from training data.
    
    These are STATIC features computed from train+valid, safe to use anytime.
    """
    edge_stats = {}
    n_train_valid = len(train_valid_timestamps)
    
    for edge, contact_dict in contact.items():
        # Get contact sequence for train+valid period
        contacts = [contact_dict.get(t, 0) for t in train_valid_timestamps]
        
        total_on = sum(contacts)
        baseline_freq = total_on / n_train_valid if n_train_valid > 0 else 0.0
        
        # Find first appearance
        first_on_idx = None
        for i, c in enumerate(contacts):
            if c == 1:
                first_on_idx = i
                break
        
        # Compute burstiness: variance of inter-contact intervals
        on_indices = [i for i, c in enumerate(contacts) if c == 1]
        if len(on_indices) > 1:
            intervals = [on_indices[i+1] - on_indices[i] for i in range(len(on_indices)-1)]
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            burstiness = std_interval / mean_interval if mean_interval > 0 else 0.0
        else:
            burstiness = 0.0
        
        # Compute flip rate (volatility)
        flip_count = sum(1 for i in range(1, len(contacts)) if contacts[i] != contacts[i-1])
        flip_rate = flip_count / (n_train_valid - 1) if n_train_valid > 1 else 0.0
        
        # Compute co-activation with neighbor edges
        node_i, node_j = edge
        neighbor_edges = (node_to_edges.get(node_i, set()) | node_to_edges.get(node_j, set())) - {edge}
        
        # Average co-activation rate: when this edge is ON, what fraction of neighbor edges are also ON?
        coactivation_rates = []
        for t_idx, t in enumerate(train_valid_timestamps):
            if contacts[t_idx] == 1:  # this edge is ON
                neighbor_on_count = 0
                for ne in neighbor_edges:
                    if contact.get(ne, {}).get(t, 0) == 1:
                        neighbor_on_count += 1
                if len(neighbor_edges) > 0:
                    coactivation_rates.append(neighbor_on_count / len(neighbor_edges))
        
        avg_coactivation = np.mean(coactivation_rates) if coactivation_rates else 0.0
        
        edge_stats[edge] = {
            'total_on_train': total_on,
            'baseline_freq': baseline_freq,
            'first_on_idx': first_on_idx if first_on_idx is not None else n_train_valid,
            'burstiness': burstiness,
            'flip_rate': flip_rate,
            'avg_coactivation': avg_coactivation,
            'n_neighbor_edges': len(neighbor_edges)
        }
    
    return edge_stats


def compute_node_historical_stats(
    adj: Dict[int, Dict[int, Set[int]]],
    train_valid_timestamps: List[int],
    all_nodes: Set[int]
) -> Dict[int, Dict[str, float]]:
    """Compute historical statistics for each node from training data."""
    node_stats = {}
    
    for node in all_nodes:
        degrees = [len(adj.get(t, {}).get(node, set())) for t in train_valid_timestamps]
        
        node_stats[node] = {
            'avg_degree': np.mean(degrees) if degrees else 0.0,
            'max_degree': max(degrees) if degrees else 0,
            'degree_std': np.std(degrees) if degrees else 0.0
        }
    
    return node_stats


# =============================================================================
# Feature Engineering
# =============================================================================

def compute_features_for_edge_at_time(
    edge: Tuple[int, int],
    time: int,
    contact_history: Dict[int, int],
    full_contact: Dict[Tuple[int, int], Dict[int, int]],
    adj: Dict[int, Dict[int, Set[int]]],
    all_timestamps: List[int],
    time_to_idx: Dict[int, int],
    edge_stats: Dict[str, float],
    node_stats: Dict[int, Dict[str, float]],
    node_to_edges: Dict[int, Set[Tuple[int, int]]],
    clip_value: int = 100
) -> np.ndarray:
    """Compute comprehensive feature vector for a single edge at a specific time.
    
    All features use ONLY data from t-1 or earlier (no leakage).
    """
    node_i, node_j = edge
    t_idx = time_to_idx[time]
    
    features = []
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_contact_at_offset(offset: int) -> int:
        if t_idx - offset < 0:
            return 0
        past_time = all_timestamps[t_idx - offset]
        return contact_history.get(past_time, 0)
    
    def get_adj_at_offset(offset: int) -> Dict[int, Set[int]]:
        if t_idx - offset < 0:
            return {}
        past_time = all_timestamps[t_idx - offset]
        return adj.get(past_time, {})
    
    def get_edge_contact_at_offset(e: Tuple[int, int], offset: int) -> int:
        if t_idx - offset < 0:
            return 0
        past_time = all_timestamps[t_idx - offset]
        return full_contact.get(e, {}).get(past_time, 0)
    
    # =========================================================================
    # 1. EXTENDED TEMPORAL HISTORY (lags 1-5)
    # =========================================================================
    y_lags = [get_contact_at_offset(i) for i in range(1, 6)]
    features.extend(y_lags)  # 5 features
    
    # =========================================================================
    # 2. RECENCY FEATURES
    # =========================================================================
    time_since_on = clip_value
    for offset in range(1, min(t_idx + 1, clip_value + 1)):
        if get_contact_at_offset(offset) == 1:
            time_since_on = offset
            break
    features.append(time_since_on)
    
    time_since_off = clip_value
    for offset in range(1, min(t_idx + 1, clip_value + 1)):
        if get_contact_at_offset(offset) == 0:
            time_since_off = offset
            break
    features.append(time_since_off)
    
    on_run_length = 0
    for offset in range(1, min(t_idx + 1, clip_value + 1)):
        if get_contact_at_offset(offset) == 1:
            on_run_length += 1
        else:
            break
    features.append(min(on_run_length, clip_value))
    
    off_run_length = 0
    for offset in range(1, min(t_idx + 1, clip_value + 1)):
        if get_contact_at_offset(offset) == 0:
            off_run_length += 1
        else:
            break
    features.append(min(off_run_length, clip_value))
    
    # =========================================================================
    # 3. MULTI-SCALE DUTY CYCLES (W5, W10, W20)
    # =========================================================================
    duty_cycles = {}
    for window in [5, 10, 20]:
        window_contacts = [get_contact_at_offset(i) for i in range(1, min(window + 1, t_idx + 1))]
        dc = sum(window_contacts) / len(window_contacts) if window_contacts else 0.0
        duty_cycles[window] = dc
        features.append(dc)
    
    # =========================================================================
    # 4. MULTI-SCALE FLIP COUNTS (W5, W10, W20)
    # =========================================================================
    for window in [5, 10, 20]:
        window_contacts = [get_contact_at_offset(i) for i in range(1, min(window + 1, t_idx + 1))]
        flip_count = sum(1 for i in range(1, len(window_contacts)) if window_contacts[i] != window_contacts[i-1]) if len(window_contacts) > 1 else 0
        features.append(flip_count)
    
    # =========================================================================
    # 5. CUMULATIVE CONTACT STATISTICS
    # =========================================================================
    total_on_so_far = sum(get_contact_at_offset(i) for i in range(1, t_idx + 1))
    features.append(min(total_on_so_far, clip_value))
    
    cumulative_freq = total_on_so_far / t_idx if t_idx > 0 else 0.0
    features.append(cumulative_freq)
    
    first_on_offset = clip_value
    for offset in range(t_idx, 0, -1):
        if get_contact_at_offset(offset) == 1:
            first_on_offset = offset
    features.append(min(first_on_offset, clip_value))
    
    # =========================================================================
    # 6. EDGE HISTORICAL STATISTICS (from train+valid, static)
    # =========================================================================
    features.append(edge_stats.get('baseline_freq', 0.0))
    features.append(min(edge_stats.get('total_on_train', 0), clip_value))
    features.append(edge_stats.get('burstiness', 0.0))
    features.append(edge_stats.get('flip_rate', 0.0))
    features.append(edge_stats.get('avg_coactivation', 0.0))
    
    # =========================================================================
    # 7. TREND / MOMENTUM FEATURES
    # =========================================================================
    dc_w5 = duty_cycles.get(5, 0.0)
    dc_w10 = duty_cycles.get(10, 0.0)
    contact_momentum = dc_w5 - dc_w10
    features.append(contact_momentum)
    
    baseline = edge_stats.get('baseline_freq', 0.0)
    recent_vs_baseline = dc_w10 / baseline if baseline > 0 else dc_w10 * 10
    features.append(min(recent_vs_baseline, 10.0))
    
    # =========================================================================
    # 8. NODE DEGREE FEATURES (from t-1, t-2)
    # =========================================================================
    adj_t1 = get_adj_at_offset(1)
    adj_t2 = get_adj_at_offset(2)
    
    deg_i_t1 = len(adj_t1.get(node_i, set()))
    deg_j_t1 = len(adj_t1.get(node_j, set()))
    deg_i_t2 = len(adj_t2.get(node_i, set()))
    deg_j_t2 = len(adj_t2.get(node_j, set()))
    
    features.extend([deg_i_t1, deg_j_t1])
    features.extend([deg_i_t1 - deg_i_t2, deg_j_t1 - deg_j_t2])
    features.extend([deg_i_t1 + deg_j_t1, deg_i_t1 * deg_j_t1])
    features.append(node_stats.get(node_i, {}).get('avg_degree', 0.0))
    features.append(node_stats.get(node_j, {}).get('avg_degree', 0.0))
    
    # =========================================================================
    # 9. GRAPH STRUCTURE FEATURES (from t-1)
    # =========================================================================
    neighbors_i = adj_t1.get(node_i, set())
    neighbors_j = adj_t1.get(node_j, set())
    common_neighbors = neighbors_i & neighbors_j
    n_common = len(common_neighbors)
    
    features.append(n_common)
    
    union_size = len(neighbors_i | neighbors_j)
    jaccard = n_common / union_size if union_size > 0 else 0.0
    features.append(jaccard)
    
    adamic_adar = sum(1.0 / math.log(len(adj_t1.get(k, set()))) for k in common_neighbors if len(adj_t1.get(k, set())) > 1)
    features.append(adamic_adar)
    
    resource_allocation = sum(1.0 / len(adj_t1.get(k, set())) for k in common_neighbors if len(adj_t1.get(k, set())) > 0)
    features.append(resource_allocation)
    
    neighbors_i_t2 = adj_t2.get(node_i, set())
    neighbors_j_t2 = adj_t2.get(node_j, set())
    common_t2 = len(neighbors_i_t2 & neighbors_j_t2)
    features.append(n_common - common_t2)
    features.append(union_size - len(neighbors_i_t2 | neighbors_j_t2))
    
    # =========================================================================
    # 10. NETWORK EFFECT: NODE ACTIVITY RATE (fraction of node's edges active at t-1)
    # =========================================================================
    # What fraction of node_i's candidate edges are active at t-1?
    edges_i = node_to_edges.get(node_i, set())
    edges_j = node_to_edges.get(node_j, set())
    
    active_edges_i = sum(1 for e in edges_i if get_edge_contact_at_offset(e, 1) == 1)
    active_edges_j = sum(1 for e in edges_j if get_edge_contact_at_offset(e, 1) == 1)
    
    activity_rate_i = active_edges_i / len(edges_i) if edges_i else 0.0
    activity_rate_j = active_edges_j / len(edges_j) if edges_j else 0.0
    
    features.append(activity_rate_i)
    features.append(activity_rate_j)
    features.append((activity_rate_i + activity_rate_j) / 2)  # avg activity
    features.append(min(activity_rate_i, activity_rate_j))  # min activity
    
    # =========================================================================
    # 11. NETWORK EFFECT: NEIGHBOR EDGE CASCADE (edges near (i,j) that changed state)
    # =========================================================================
    # Neighbor edges = edges incident to i or j, excluding (i,j) itself
    neighbor_edges = (edges_i | edges_j) - {edge}
    n_neighbor_edges = len(neighbor_edges)
    
    # Count edges that turned ON in last step (t-2 OFF, t-1 ON)
    turned_on = sum(1 for e in neighbor_edges 
                    if get_edge_contact_at_offset(e, 1) == 1 and get_edge_contact_at_offset(e, 2) == 0)
    # Count edges that turned OFF in last step
    turned_off = sum(1 for e in neighbor_edges
                     if get_edge_contact_at_offset(e, 1) == 0 and get_edge_contact_at_offset(e, 2) == 1)
    
    features.append(turned_on)
    features.append(turned_off)
    features.append(turned_on / n_neighbor_edges if n_neighbor_edges > 0 else 0.0)  # fraction turned on
    features.append(turned_off / n_neighbor_edges if n_neighbor_edges > 0 else 0.0)  # fraction turned off
    
    # Count edges that are currently ON at t-1
    neighbor_on_count = sum(1 for e in neighbor_edges if get_edge_contact_at_offset(e, 1) == 1)
    features.append(neighbor_on_count)
    features.append(neighbor_on_count / n_neighbor_edges if n_neighbor_edges > 0 else 0.0)
    
    # =========================================================================
    # 12. NETWORK EFFECT: LOCAL CLUSTERING COEFFICIENT (at t-1)
    # =========================================================================
    # Clustering coefficient for node i: fraction of pairs of i's neighbors that are connected
    def clustering_coefficient(node: int, adj_t: Dict[int, Set[int]]) -> float:
        neighbors = adj_t.get(node, set())
        k = len(neighbors)
        if k < 2:
            return 0.0
        # Count edges between neighbors
        neighbor_edges_count = 0
        neighbors_list = list(neighbors)
        for idx1 in range(len(neighbors_list)):
            for idx2 in range(idx1 + 1, len(neighbors_list)):
                n1, n2 = neighbors_list[idx1], neighbors_list[idx2]
                if n2 in adj_t.get(n1, set()):
                    neighbor_edges_count += 1
        max_edges = k * (k - 1) / 2
        return neighbor_edges_count / max_edges if max_edges > 0 else 0.0
    
    cc_i = clustering_coefficient(node_i, adj_t1)
    cc_j = clustering_coefficient(node_j, adj_t1)
    features.append(cc_i)
    features.append(cc_j)
    features.append((cc_i + cc_j) / 2)  # average clustering
    
    # =========================================================================
    # 13. NETWORK EFFECT: TRIANGLE PARTICIPATION (at t-1)
    # =========================================================================
    # Number of triangles this edge participates in = number of common neighbors
    # (already captured above, but let's add triangle-based features)
    triangles = n_common  # each common neighbor forms a triangle
    features.append(triangles)
    
    # Potential triangles: if this edge is active, how many triangles COULD form?
    # = number of paths of length 2 between i and j at t-1 (already counted as common neighbors)
    potential_triangles = n_common
    features.append(potential_triangles)
    
    # =========================================================================
    # 14. NETWORK EFFECT: NEIGHBOR EDGE STABILITY (from history)
    # =========================================================================
    # Average duty cycle of neighbor edges (over last W10)
    neighbor_duty_cycles = []
    for ne in neighbor_edges:
        ne_contacts = [get_edge_contact_at_offset(ne, i) for i in range(1, min(11, t_idx + 1))]
        if ne_contacts:
            neighbor_duty_cycles.append(sum(ne_contacts) / len(ne_contacts))
    
    avg_neighbor_duty_cycle = np.mean(neighbor_duty_cycles) if neighbor_duty_cycles else 0.0
    std_neighbor_duty_cycle = np.std(neighbor_duty_cycles) if len(neighbor_duty_cycles) > 1 else 0.0
    features.append(avg_neighbor_duty_cycle)
    features.append(std_neighbor_duty_cycle)
    
    # =========================================================================
    # 15. NETWORK EFFECT: COMMON NEIGHBOR ACTIVITY
    # =========================================================================
    # Among common neighbors, how many have edges to BOTH i and j that are active at t-1?
    common_neighbor_both_active = 0
    for k in common_neighbors:
        edge_ik = (min(node_i, k), max(node_i, k))
        edge_jk = (min(node_j, k), max(node_j, k))
        if get_edge_contact_at_offset(edge_ik, 1) == 1 and get_edge_contact_at_offset(edge_jk, 1) == 1:
            common_neighbor_both_active += 1
    
    features.append(common_neighbor_both_active)
    features.append(common_neighbor_both_active / n_common if n_common > 0 else 0.0)
    
    # =========================================================================
    # 16. NETWORK EFFECT: EDGE SYNCHRONY WITH NEIGHBORS
    # =========================================================================
    # How often does this edge match the majority state of its neighbors over last W5?
    synchrony_count = 0
    for offset in range(1, min(6, t_idx + 1)):
        this_state = get_contact_at_offset(offset)
        neighbor_states = [get_edge_contact_at_offset(ne, offset) for ne in neighbor_edges]
        if neighbor_states:
            majority = 1 if sum(neighbor_states) > len(neighbor_states) / 2 else 0
            if this_state == majority:
                synchrony_count += 1
    
    synchrony_rate = synchrony_count / min(5, t_idx) if t_idx > 0 else 0.0
    features.append(synchrony_rate)
    
    return np.array(features, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Return names of all features in order."""
    names = []
    
    # 1. Extended temporal history (5)
    names.extend([f'y_t_minus_{i}' for i in range(1, 6)])
    
    # 2. Recency features (4)
    names.extend(['time_since_on', 'time_since_off', 'on_run_length', 'off_run_length'])
    
    # 3. Multi-scale duty cycles (3)
    names.extend(['duty_cycle_W5', 'duty_cycle_W10', 'duty_cycle_W20'])
    
    # 4. Multi-scale flip counts (3)
    names.extend(['flip_count_W5', 'flip_count_W10', 'flip_count_W20'])
    
    # 5. Cumulative contact statistics (3)
    names.extend(['total_on_so_far', 'cumulative_freq', 'time_since_first_on'])
    
    # 6. Edge historical statistics (5)
    names.extend(['baseline_freq', 'total_on_train', 'burstiness', 'flip_rate', 'avg_coactivation'])
    
    # 7. Trend/momentum features (2)
    names.extend(['contact_momentum', 'recent_vs_baseline'])
    
    # 8. Node degree features (8)
    names.extend(['deg_i_t1', 'deg_j_t1', 'deg_i_change', 'deg_j_change',
                  'deg_sum', 'deg_product', 'deg_i_avg', 'deg_j_avg'])
    
    # 9. Graph structure features (6)
    names.extend(['common_neighbors', 'jaccard_coefficient', 'adamic_adar_index', 
                  'resource_allocation_index', 'common_neighbors_change', 'union_size_change'])
    
    # 10. Node activity rate (4)
    names.extend(['activity_rate_i', 'activity_rate_j', 'avg_activity_rate', 'min_activity_rate'])
    
    # 11. Neighbor edge cascade (6)
    names.extend(['neighbor_turned_on', 'neighbor_turned_off', 'frac_neighbor_turned_on',
                  'frac_neighbor_turned_off', 'neighbor_on_count', 'frac_neighbor_on'])
    
    # 12. Local clustering (3)
    names.extend(['clustering_coef_i', 'clustering_coef_j', 'avg_clustering_coef'])
    
    # 13. Triangle participation (2)
    names.extend(['triangles', 'potential_triangles'])
    
    # 14. Neighbor edge stability (2)
    names.extend(['avg_neighbor_duty_cycle', 'std_neighbor_duty_cycle'])
    
    # 15. Common neighbor activity (2)
    names.extend(['common_neighbor_both_active', 'frac_common_neighbor_both_active'])
    
    # 16. Edge synchrony (1)
    names.extend(['synchrony_rate'])
    
    return names


# =============================================================================
# Dataset Building
# =============================================================================

def build_training_dataset(
    train_quads: List[Quadruple],
    valid_quads: List[Quadruple],
    candidate_edges: Set[Tuple[int, int]],
    train_timestamps: List[int],
    valid_timestamps: List[int],
    all_timestamps: List[int]
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """Build feature matrix and labels for training."""
    all_train_valid = train_quads + valid_quads
    train_valid_timestamps = sorted(set(train_timestamps + valid_timestamps))
    
    contact = build_contact_matrix(all_train_valid, candidate_edges, all_timestamps)
    adj = build_adjacency_at_time(all_train_valid, all_timestamps)
    node_to_edges = build_node_to_edges(candidate_edges)
    
    all_nodes = set()
    for edge in candidate_edges:
        all_nodes.add(edge[0])
        all_nodes.add(edge[1])
    
    edge_stats = compute_edge_historical_stats(contact, train_valid_timestamps, node_to_edges)
    node_stats = compute_node_historical_stats(adj, train_valid_timestamps, all_nodes)
    
    time_to_idx = {t: i for i, t in enumerate(all_timestamps)}
    
    X_list = []
    y_list = []
    
    for edge in candidate_edges:
        contact_history = contact[edge]
        e_stats = edge_stats.get(edge, {})
        
        for t in train_valid_timestamps:
            t_idx = time_to_idx[t]
            if t_idx < 1:
                continue
            
            features = compute_features_for_edge_at_time(
                edge=edge,
                time=t,
                contact_history=contact_history,
                full_contact=contact,
                adj=adj,
                all_timestamps=all_timestamps,
                time_to_idx=time_to_idx,
                edge_stats=e_stats,
                node_stats=node_stats,
                node_to_edges=node_to_edges
            )
            
            label = contact_history[t]
            X_list.append(features)
            y_list.append(label)
    
    return np.array(X_list), np.array(y_list), edge_stats, node_stats, node_to_edges


def build_test_dataset_autoregressive(
    train_quads: List[Quadruple],
    valid_quads: List[Quadruple],
    candidate_edges: Set[Tuple[int, int]],
    test_timestamps: List[int],
    all_timestamps: List[int],
    model: xgb.XGBClassifier,
    edge_stats: Dict,
    node_stats: Dict,
    node_to_edges: Dict,
    threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Build test predictions autoregressively."""
    all_train_valid = train_quads + valid_quads
    
    contact = build_contact_matrix(all_train_valid, candidate_edges, all_timestamps)
    adj = build_adjacency_at_time(all_train_valid, all_timestamps)
    time_to_idx = {t: i for i, t in enumerate(all_timestamps)}
    
    predictions = []
    y_pred_all = []
    
    for t in sorted(test_timestamps):
        t_idx = time_to_idx.get(t)
        if t_idx is None or t_idx < 1:
            continue
        
        X_t = []
        edges_at_t = []
        
        for edge in candidate_edges:
            contact_history = contact[edge]
            e_stats = edge_stats.get(edge, {})
            
            features = compute_features_for_edge_at_time(
                edge=edge,
                time=t,
                contact_history=contact_history,
                full_contact=contact,
                adj=adj,
                all_timestamps=all_timestamps,
                time_to_idx=time_to_idx,
                edge_stats=e_stats,
                node_stats=node_stats,
                node_to_edges=node_to_edges
            )
            
            X_t.append(features)
            edges_at_t.append(edge)
        
        if len(X_t) == 0:
            continue
        
        X_t = np.array(X_t)
        y_proba_t = model.predict_proba(X_t)[:, 1]
        y_pred_t = (y_proba_t >= threshold).astype(int)
        
        predictions_at_t = []
        
        for i, edge in enumerate(edges_at_t):
            pred = y_pred_t[i]
            y_pred_all.append(pred)
            contact[edge][t] = pred
            
            if pred == 1:
                node_i, node_j = edge
                predictions.append((node_i, node_j, t))
                predictions_at_t.append((node_i, node_j))
        
        update_adjacency_with_predictions(adj, predictions_at_t, t)
    
    return predictions, np.array(y_pred_all), None


# =============================================================================
# Training and Prediction
# =============================================================================

def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"Training samples: {len(y_train)}")
    print(f"  Positive samples: {n_pos} ({100*n_pos/len(y_train):.1f}%)")
    print(f"  Negative samples: {n_neg} ({100*n_neg/len(y_train):.1f}%)")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    print(f"\nFeature Importance (top 20 of {len(get_feature_names())}):")
    feature_names = get_feature_names()
    importance = model.feature_importances_
    sorted_features = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    for name, imp in sorted_features[:20]:
        print(f"  {name}: {imp:.4f}")
    
    return model


# =============================================================================
# Output and Evaluation
# =============================================================================

def write_predictions(predictions: List[Tuple[int, int, int]], output_path: str) -> None:
    """Write predictions in RE-Net compatible format."""
    predictions_sorted = sorted(predictions, key=lambda x: (x[2], x[0], x[1]))
    with open(output_path, 'w') as f:
        for subj, obj, time in predictions_sorted:
            f.write(f"{subj:4d} {obj:4d} {time:4d}\n")
    print(f"\nWrote {len(predictions)} predictions to {output_path}")


def evaluate_autoregressive_predictions(
    predictions: List[Tuple[int, int, int]],
    test_quads: List[Quadruple],
    candidate_edges: Set[Tuple[int, int]],
    test_timestamps: List[int]
) -> None:
    """Evaluate autoregressive predictions against ground truth."""
    gt_set = set()
    for q in test_quads:
        edge = (min(q.subject, q.obj), max(q.subject, q.obj))
        gt_set.add((edge, q.time))
    
    pred_set = set()
    for subj, obj, t in predictions:
        edge = (min(subj, obj), max(subj, obj))
        pred_set.add((edge, t))
    
    n_samples = len(candidate_edges) * len(test_timestamps)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    tn = n_samples - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("Autoregressive Test Evaluation (Strict Holdout)")
    print("=" * 50)
    print(f"Total test samples (edge × time): {n_samples}")
    print(f"Ground truth positives: {len(gt_set)}")
    print(f"Predicted positives: {len(pred_set)}")
    print()
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("=" * 50)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost POC with Network Effect Features (Autoregressive)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--evaluate', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost POC with Network Effect Features")
    print("(Autoregressive / Strict Holdout Forecasting)")
    print("=" * 60)
    
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from: {data_dir}")
    
    train_quads = load_quadruples(str(data_dir / 'train.txt'))
    valid_quads = load_quadruples(str(data_dir / 'valid.txt'))
    test_quads = load_quadruples(str(data_dir / 'test.txt'))
    num_entities, num_relations = load_stat(str(data_dir / 'stat.txt'))
    
    print(f"  Train: {len(train_quads)} | Valid: {len(valid_quads)} | Test: {len(test_quads)}")
    print(f"  Entities: {num_entities}, Relations: {num_relations}")
    
    print("\nIdentifying candidate edges (train+valid only)...")
    train_valid_quads = train_quads + valid_quads
    candidate_edges = get_candidate_edges([train_valid_quads])
    
    all_quads = train_quads + valid_quads + test_quads
    all_timestamps = get_all_timestamps([all_quads])
    
    train_timestamps = sorted(set(q.time for q in train_quads))
    valid_timestamps = sorted(set(q.time for q in valid_quads))
    test_timestamps = sorted(set(q.time for q in test_quads))
    
    test_edges = get_candidate_edges([test_quads])
    novel_test_edges = test_edges - candidate_edges
    
    print(f"  Candidate edges: {len(candidate_edges)}")
    print(f"  Novel test edges: {len(novel_test_edges)}")
    print(f"  Timestamps - Train: {len(train_timestamps)} | Valid: {len(valid_timestamps)} | Test: {len(test_timestamps)}")
    
    print("\nBuilding training dataset with network effect features...")
    X_train, y_train, edge_stats, node_stats, node_to_edges = build_training_dataset(
        train_quads=train_quads,
        valid_quads=valid_quads,
        candidate_edges=candidate_edges,
        train_timestamps=train_timestamps,
        valid_timestamps=valid_timestamps,
        all_timestamps=all_timestamps
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features per sample: {X_train.shape[1]}")
    
    print("\nTraining XGBoost model...")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state
    )
    
    print("\nRunning autoregressive test prediction...")
    predictions, y_pred_all, _ = build_test_dataset_autoregressive(
        train_quads=train_quads,
        valid_quads=valid_quads,
        candidate_edges=candidate_edges,
        test_timestamps=test_timestamps,
        all_timestamps=all_timestamps,
        model=model,
        edge_stats=edge_stats,
        node_stats=node_stats,
        node_to_edges=node_to_edges,
        threshold=args.threshold
    )
    
    gt_on_candidate = sum(
        1 for q in test_quads 
        if (min(q.subject, q.obj), max(q.subject, q.obj)) in candidate_edges
    )
    
    print(f"  Predicted contacts: {len(predictions)}")
    print(f"  Ground truth on candidate edges: {gt_on_candidate}")
    
    if args.evaluate:
        evaluate_autoregressive_predictions(
            predictions=predictions,
            test_quads=test_quads,
            candidate_edges=candidate_edges,
            test_timestamps=test_timestamps
        )
    
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    write_predictions(predictions, args.output_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
