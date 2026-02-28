import os

# =============================
# LABEL MAPPING
# =============================

LABEL_MAP = {
    "normal": 0,
    "scratch": 1,
    "crack": 2,
    "dent": 3,
    "pit": 4,
    "rust": 5,
    "stain": 6,
}

def get_label_from_folder(folder_name):
    """
    Convert folder name to numeric label
    """
    folder_name = folder_name.lower()
    if folder_name not in LABEL_MAP:
        raise ValueError(f"Unknown class: {folder_name}")
    return LABEL_MAP[folder_name]


if __name__ == "__main__":
    print("Label mapping loaded successfully!")