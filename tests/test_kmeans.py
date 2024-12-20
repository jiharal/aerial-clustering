import pytest
import numpy as np
from datetime import datetime
from aerial_clustering.kmeans import KMeansClustering

@pytest.fixture
def sample_data():
    """Fixture for sample test data"""
    return {
        "aerial_log": {
            "heatmap": [
                [-2.968123, 104.80019899999999, "07 Dec 2024 18:09:17 WIB", ["07 Dec 2024 18:09:17 WIB"]],
                [-2.96525, 104.79828, "07 Dec 2024 18:09:18 WIB", ["07 Dec 2024 18:09:18 WIB"]],
                [-2.972362, 104.791319, "07 Dec 2024 18:10:19 WIB", ["07 Dec 2024 18:10:19 WIB"]],
                [-2.96525, 104.79828, "07 Dec 2024 18:10:20 WIB", ["07 Dec 2024 18:10:20 WIB"]]
            ]
        }
    }

def test_initialization():
    """Test KMeansClustering initialization"""
    kmeans = KMeansClustering(max_distance=100)
    assert kmeans.max_distance == 100
    assert isinstance(kmeans.clusters, list)
    assert len(kmeans.clusters) == 0

def test_haversine_distance():
    """Test haversine distance calculation"""
    kmeans = KMeansClustering()
    point1 = np.array([-2.968123, 104.800199])
    point2 = np.array([-2.96525, 104.79828])
    
    distance = kmeans.haversine_distance(point1, point2)
    assert isinstance(distance, float)
    assert distance > 0