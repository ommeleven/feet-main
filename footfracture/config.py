"""Central path configuration.

The MRI dataset is not distributed with this repository (see docs/DATA.md).
Point FOOT_FRACTURE_DATA_DIR at your local copy; defaults to ./data.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("FOOT_FRACTURE_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RESULTS_DIR = os.environ.get("FOOT_FRACTURE_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))

# Label Studio annotation CSV used by several preprocessing / bbox scripts
ANNOTATIONS_CSV = os.path.join(
    DATA_DIR, "json_data", "sag_T1_json_columns", "annotations", "results_box_cleaned_calculation.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
