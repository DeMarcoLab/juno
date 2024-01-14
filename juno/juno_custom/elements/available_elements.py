import os

CONFIG_PATH = __file__
BASE_PATH = os.path.dirname(CONFIG_PATH)
# LOG_PATH = os.path.join(BASE_PATH, 'log')

# os.makedirs(LOG_PATH, exist_ok=True)

# structure is {config_name: [folder_name.file_name, class_name]}
AVAILABLE_ELEMENTS = {
    "herschel": ["Herschel.herschel", "Herschel"],
    "lattice": ["Lattice.lattice", "Lattice"],
    "microlens": ["Microlens.microlens", "ExtendedMicrolens"],
}