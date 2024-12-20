# Aerial Clustering

A Python library for clustering aerial data points based on distance.

## Installation

```bash
pip install aerial-clustering
```

## Usage

```python
from aerial_clustering import KMeansClustering, read_json_data

# Read data
data = read_json_data("path/to/data.json")

# Process data with 100m radius
clustering = KMeansClustering(max_distance=100)
results = clustering.process_aerial_data(data)

# Print results
for cluster in results:
    print(f"Latitude: {cluster['latitude']}")
    print(f"Longitude: {cluster['longitude']}")
    print(f"Timestamp: {cluster['timestamp']}")
    print(f"Points in radius: {cluster['point_count']}")
```
