"""
Helper utility functions
"""
import hashlib
import numpy as np
from typing import Dict, List
from datetime import datetime


def deterministic_random(seed_str: str, low: float, high: float) -> float:
    """Generate deterministic pseudo-random number"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)


def robust_normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize single value to [0, 1] range"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    u = np.array(vec1)
    v = np.array(vec2)
    
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return float(dot_product / (norm_u * norm_v))


def add_log(admin_logs: List[Dict], action: str, status: str, details: str):
    """Add admin log"""
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "status": status,
        "details": details
    }
    admin_logs.append(log)
    if len(admin_logs) > 100:
        admin_logs.pop(0)

