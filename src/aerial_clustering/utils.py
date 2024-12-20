import json
from typing import Dict


def read_json_data(file_path: str = "data/data.json") -> Dict:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error membaca file: {str(e)}")
        return None
