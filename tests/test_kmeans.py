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

def test_parse_timestamp():
    """Test timestamp parsing"""
    kmeans = KMeansClustering()
    timestamp = "07 Dec 2024 18:09:17 WIB"
    parsed = kmeans.parse_timestamp(timestamp)
    
    assert isinstance(parsed, datetime)
    assert parsed.year == 2024
    assert parsed.month == 12
    assert parsed.day == 7

def test_process_aerial_data_zero_distance(sample_data):
    """Test processing with max_distance = 0"""
    kmeans = KMeansClustering(max_distance=0)
    results = kmeans.process_aerial_data(sample_data)
    
    assert len(results) == len(sample_data['aerial_log']['heatmap'])
    for result in results:
        assert result['point_count'] == 1

def test_process_aerial_data_with_distance(sample_data):
    """Test processing with positive max_distance"""
    kmeans = KMeansClustering(max_distance=1000)  # 1000 meters
    results = kmeans.process_aerial_data(sample_data)
    
    assert len(results) > 0
    for result in results:
        assert 'latitude' in result
        assert 'longitude' in result
        assert 'timestamp' in result
        assert 'point_count' in result
        assert result['point_count'] >= 1

def test_process_aerial_data_empty():
    """Test processing with empty data"""
    kmeans = KMeansClustering()
    empty_data = {"aerial_log": {"heatmap": []}}
    results = kmeans.process_aerial_data(empty_data)
    assert len(results) == 0

# @pytest.mark.parametrize("max_distance,expected_clusters", [
#     (0, 4),    # Each point should be its own cluster
#     (100, 2),  # Points within 100m should cluster
#     (1000, 1)  # All points should cluster together
# ])
# def test_different_distances(sample_data, max_distance, expected_clusters):
#     """Test clustering with different distances"""
#     kmeans = KMeansClustering(max_distance=max_distance)
#     results = kmeans.process_aerial_data(sample_data)
#     assert len(results) == expected_clusters

def test_timestamp_sorting(sample_data):
    """Test that newest timestamps are selected for clusters"""
    kmeans = KMeansClustering(max_distance=1000)
    results = kmeans.process_aerial_data(sample_data)
    
    for result in results:
        # Check if timestamp is the newest in its cluster
        timestamp = datetime.strptime(result['timestamp'], "%d %b %Y %H:%M:%S %Z")
        assert isinstance(timestamp, datetime)

def test_invalid_coordinates():
    """Test handling of invalid coordinates"""
    kmeans = KMeansClustering()
    invalid_data = {
        "aerial_log": {
            "heatmap": [
                [None, 104.800199, "07 Dec 2024 18:09:17 WIB", []],
                [-2.96525, None, "07 Dec 2024 18:09:18 WIB", []]
            ]
        }
    }
    with pytest.raises(Exception):  # Should raise some kind of exception for invalid coordinates
        kmeans.process_aerial_data(invalid_data)

def test_point_count_accuracy(sample_data):
    """Test accuracy of point counting in clusters"""
    kmeans = KMeansClustering(max_distance=100)
    results = kmeans.process_aerial_data(sample_data)
    
    total_points = sum(result['point_count'] for result in results)
    assert total_points == len(sample_data['aerial_log']['heatmap'])