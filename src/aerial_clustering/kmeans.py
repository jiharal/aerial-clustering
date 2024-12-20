from datetime import datetime
from typing import Dict, List

import numpy as np


class KMeansClustering:
    def __init__(self, max_distance: float = 100):
        """
        Inisialisasi clustering berdasarkan radius
        Parameters:
        -----------
        max_distance : float
            Radius maksimum dalam meter
        """
        self.max_distance = max_distance
        self.clusters = []  # Untuk menyimpan hasil clustering

    def haversine_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Menghitung jarak Haversine antara dua titik koordinat
        Returns jarak dalam meter
        """
        R = 6371000

        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp dari string"""
        return datetime.strptime(timestamp_str, "%d %b %Y %H:%M:%S %Z")

    def process_aerial_data(self, data: Dict) -> List[Dict]:
        """
        Memproses data aerial dan mengelompokkan berdasarkan radius
        Parameters:
        -----------
        data : Dict
            Dictionary berisi data aerial_log
        Returns:
        --------
        List[Dict]
            List berisi data yang telah dikelompokkan
        """
        points = []
        # Ekstrak dan konversi data
        for point in data["aerial_log"]["heatmap"]:
            lat, lon, timestamp, _ = point
            points.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": timestamp,
                    "coords": np.array([lat, lon]),
                    "datetime": self.parse_timestamp(timestamp),
                }
            )

        # Jika max_distance adalah 0, kembalikan semua point
        if self.max_distance == 0:
            return [
                {
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                    "timestamp": point["timestamp"],
                    "point_count": 1,
                }
                for point in points
            ]

        # Urutkan berdasarkan timestamp terbaru
        points.sort(key=lambda x: x["datetime"], reverse=True)

        processed_clusters = []
        used_points = set()

        # Proses setiap point
        for i, point in enumerate(points):
            if i in used_points:
                continue

            cluster = [point]
            used_points.add(i)

            # Cek point lain yang dalam radius
            for j, other_point in enumerate(points):
                if j in used_points:
                    continue

                distance = self.haversine_distance(
                    point["coords"], other_point["coords"]
                )
                # Tambahkan epsilon untuk mengatasi masalah floating point
                if distance <= (self.max_distance + 1e-10):
                    cluster.append(other_point)
                    used_points.add(j)

            # Ambil data terbaru dari cluster
            newest_point = max(cluster, key=lambda x: x["datetime"])
            processed_clusters.append(
                {
                    "latitude": newest_point["latitude"],
                    "longitude": newest_point["longitude"],
                    "timestamp": newest_point["timestamp"],
                    "point_count": len(cluster),
                }
            )

        return processed_clusters
