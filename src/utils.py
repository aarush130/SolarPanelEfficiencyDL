"""
Utility Functions
=================
Helper functions for the solar panel efficiency prediction project.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def setup_logging(log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_dir : str
        Directory for log files
    level : int
        Logging level
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def create_dirs(dirs: list) -> None:
    """Create multiple directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def calculate_efficiency_stats(efficiencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for efficiency values.
    
    Parameters:
    -----------
    efficiencies : np.ndarray
        Array of efficiency values
        
    Returns:
    --------
    dict
        Dictionary containing statistics
    """
    return {
        'mean': float(np.mean(efficiencies)),
        'std': float(np.std(efficiencies)),
        'min': float(np.min(efficiencies)),
        'max': float(np.max(efficiencies)),
        'median': float(np.median(efficiencies)),
        'q25': float(np.percentile(efficiencies, 25)),
        'q75': float(np.percentile(efficiencies, 75))
    }


def efficiency_rating(efficiency: float) -> tuple:
    """
    Get efficiency rating and color.
    
    Parameters:
    -----------
    efficiency : float
        Efficiency value (0-25%)
        
    Returns:
    --------
    tuple
        (rating_text, color_hex)
    """
    if efficiency >= 20:
        return ('Excellent', '#11998e')
    elif efficiency >= 17:
        return ('Very Good', '#27ae60')
    elif efficiency >= 14:
        return ('Good', '#3498db')
    elif efficiency >= 11:
        return ('Fair', '#f39c12')
    elif efficiency >= 8:
        return ('Poor', '#e67e22')
    else:
        return ('Very Poor', '#e74c3c')


def print_banner(title: str, width: int = 60) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Update metrics for an epoch."""
        entry = {'epoch': epoch, **metrics}
        self.history.append(entry)
        self.metrics = metrics
    
    def get_best(self, metric: str, mode: str = 'min') -> Dict[str, Any]:
        """Get the epoch with best metric value."""
        if not self.history:
            return {}
        
        if mode == 'min':
            best = min(self.history, key=lambda x: x.get(metric, float('inf')))
        else:
            best = max(self.history, key=lambda x: x.get(metric, float('-inf')))
        
        return best
    
    def save(self, filepath: str) -> None:
        """Save metrics history to file."""
        save_json({'history': self.history, 'final_metrics': self.metrics}, filepath)


if __name__ == "__main__":
    # Test utilities
    print_banner("Utility Functions Test")
    
    # Test efficiency rating
    test_efficiencies = [5, 10, 13, 16, 19, 22]
    for eff in test_efficiencies:
        rating, color = efficiency_rating(eff)
        print(f"Efficiency {eff}%: {rating} ({color})")
    
    # Test stats calculation
    data = np.random.normal(15, 3, 1000)
    stats = calculate_efficiency_stats(data)
    print(f"\nSample statistics: {stats}")
