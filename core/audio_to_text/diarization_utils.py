"""
Diarization utility functions for speaker segmentation and labeling.

This module contains advanced diarization algorithms including:
- Overlapping window generation
- Segment merging
- Viterbi-based speaker label refinement
"""

from typing import List, Dict, Tuple
import numpy as np
from log_utils import setup_logger

logger = setup_logger(__name__)


def make_overlapping_windows(
    segments: List[Dict[str, float]],
    window_size: float = 0.9,
    hop_size: float = 0.25,
    min_length: float = 0.8
) -> List[Dict[str, float]]:
    """
    Generate overlapping windows of fixed length from audio segments.
    
    Windows shorter than min_length are not created to ensure meaningful segments.
    
    Args:
        segments: List of segments with 'start' and 'end' times
        window_size: Fixed window length in seconds (default: 0.9)
        hop_size: Step size between windows in seconds (default: 0.25)
        min_length: Minimum segment length to process in seconds (default: 0.8)
        
    Returns:
        List of windowed segments with 'start' and 'end' times
        
    Example:
        >>> segments = [{'start': 0.0, 'end': 2.5}]
        >>> windows = make_overlapping_windows(segments, win=0.9, hop=0.25)
        >>> # Returns: [{'start': 0.0, 'end': 0.9}, {'start': 0.25, 'end': 1.15}, ...]
    """
    out = []
    window_size = float(window_size)
    hop_size = float(hop_size)
    min_length = float(min_length)
    
    for seg in segments:
        start_time = float(seg['start'])
        end_time = float(seg['end'])
        duration = end_time - start_time
        
        # Skip segments that are too short
        if duration < min_length - 1e-6:
            logger.debug(f"Skipping segment [{start_time:.2f}, {end_time:.2f}] - too short ({duration:.2f}s)")
            continue
        
        # If segment fits within one window, use it as-is
        if duration <= window_size + 1e-6:
            out.append({'start': start_time, 'end': end_time})
            continue
        
        # Generate overlapping windows
        current_time = start_time
        while current_time + window_size <= end_time + 1e-9:
            out.append({
                'start': current_time,
                'end': min(current_time + window_size, end_time)
            })
            current_time += hop_size
    
    logger.debug(f"Generated {len(out)} windows from {len(segments)} segments")
    return out


def merge_labeled_windows(
    windows: List[Dict[str, float]],
    labels: np.ndarray,
    max_gap: float = 0.15,
    min_length: float = 0.8
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Merge consecutive windows with the same speaker label.
    
    Segments with the same label and gaps smaller than max_gap are merged.
    Short segments below min_length are filtered out.
    
    Args:
        windows: List of window segments with 'start' and 'end' times
        labels: Array of speaker labels for each window
        max_gap: Maximum gap in seconds to merge segments (default: 0.15)
        min_length: Minimum segment length to keep in seconds (default: 0.8)
        
    Returns:
        Tuple of (merged_segments, merged_labels)
        
    Example:
        >>> windows = [{'start': 0.0, 'end': 0.9}, {'start': 0.8, 'end': 1.7}]
        >>> labels = np.array([0, 0])
        >>> merged_segs, merged_labs = merge_labeled_windows(windows, labels)
        >>> # Returns merged segment if gap < max_gap
    """
    if not windows or len(windows) != len(labels):
        logger.warning(f"Invalid input: windows={len(windows)}, labels={len(labels)}")
        return [], np.array([], dtype=int)
    
    # Sort windows by start time
    items = sorted(
        [{'start': float(w['start']), 'end': float(w['end']), 'lab': int(l)}
         for w, l in zip(windows, labels)],
        key=lambda z: (z['start'], z['end'])
    )
    
    # Merge consecutive segments with same label
    merged = [{
        'start': items[0]['start'],
        'end': items[0]['end'],
        'lab': items[0]['lab']
    }]
    
    for item in items[1:]:
        current = merged[-1]
        gap = item['start'] - current['end']
        
        # Merge if same label and gap is small enough
        if item['lab'] == current['lab'] and gap <= max_gap + 1e-9:
            current['end'] = max(current['end'], item['end'])
        else:
            merged.append({
                'start': item['start'],
                'end': item['end'],
                'lab': item['lab']
            })
    
    # Filter out short segments
    kept = [m for m in merged if (m['end'] - m['start']) >= (min_length - 1e-6)]
    
    if not kept:
        logger.warning("All segments filtered out after merging")
        return [], np.array([], dtype=int)
    
    segments = [{'start': k['start'], 'end': k['end']} for k in kept]
    labels_out = np.array([k['lab'] for k in kept], dtype=int)
    
    logger.debug(f"Merged {len(windows)} windows into {len(segments)} segments")
    return segments, labels_out


def viterbi_labels_by_centroids(
    embeddings: np.ndarray,
    labels_init: np.ndarray,
    n_clusters: int,
    switch_penalty: float = 0.18
) -> np.ndarray:
    """
    Refine speaker labels using Viterbi algorithm with centroid-based costs.
    
    This algorithm uses cosine similarity to centroids with a penalty for
    speaker switches to produce smoother, more consistent speaker assignments.
    
    Args:
        embeddings: [N, d] L2-normalized embedding vectors for windows
        labels_init: Initial speaker labels from clustering
        n_clusters: Number of speaker clusters (typically 2)
        switch_penalty: Penalty for switching speakers (0.15-0.25 recommended)
        
    Returns:
        Refined speaker labels as numpy array
        
    Notes:
        - Embeddings should be L2-normalized before calling
        - Higher switch_penalty produces fewer speaker switches
        - Uses cosine similarity (dot product of normalized vectors)
        
    Algorithm:
        1. Compute speaker centroids from initial labels
        2. Calculate emission costs as -cosine_similarity
        3. Run Viterbi dynamic programming with switch penalties
        4. Backtrack to get optimal label sequence
    """
    if len(embeddings) != len(labels_init):
        raise ValueError(f"Embeddings ({len(embeddings)}) and labels ({len(labels_init)}) length mismatch")
    
    # 1) Compute centroids for each cluster
    centroids = []
    for k in range(n_clusters):
        cluster_indices = np.where(labels_init == k)[0]
        
        # Handle empty clusters (rare but possible)
        if len(cluster_indices) == 0:
            logger.warning(f"Empty cluster {k}, using most distant point")
            # Use the point farthest from the mean
            distances = np.linalg.norm(
                embeddings - embeddings.mean(0, keepdims=True),
                axis=1
            )
            cluster_indices = np.array([np.argmax(distances)])
        
        # Compute and normalize centroid
        centroid = embeddings[cluster_indices].mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)
    
    centroids = np.stack(centroids, axis=0)  # [K, d]
    
    # 2) Emission costs = -cosine_similarity (minimize)
    # Since embeddings and centroids are L2-normalized, dot product = cosine similarity
    similarities = embeddings @ centroids.T  # [N, K]
    costs = -similarities  # Lower cost = better match
    
    # 3) Viterbi dynamic programming
    N, K = costs.shape
    dp = np.zeros((N, K), dtype=np.float32)
    backtrack = np.zeros((N, K), dtype=np.int32)
    
    # Initialize first state
    dp[0] = costs[0]
    
    # Forward pass with switch penalties
    for i in range(1, N):
        # Compute cost of coming from each previous state
        # Add penalty for switching (diagonal = 0 penalty, off-diagonal = switch_penalty)
        prev_costs = dp[i-1][:, None] + switch_penalty * (1.0 - np.eye(K, dtype=np.float32))
        
        # Track best previous state for each current state
        backtrack[i] = prev_costs.argmin(axis=0)
        
        # Update DP table
        dp[i] = costs[i] + prev_costs.min(axis=0)
    
    # 4) Backward pass to recover optimal path
    labels_refined = np.zeros(N, dtype=np.int32)
    labels_refined[-1] = dp[-1].argmin()
    
    for i in range(N - 2, -1, -1):
        labels_refined[i] = backtrack[i + 1, labels_refined[i + 1]]
    
    # Log statistics
    switches = np.sum(labels_refined[1:] != labels_refined[:-1])
    logger.debug(f"Viterbi refinement: {switches} speaker switches in {N} windows")
    
    return labels_refined

