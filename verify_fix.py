
from nuscenes.eval.detection.config import config_factory
import pickle

try:
    cfg = config_factory('detection_cvpr_2019')
    print(f"Original class_names type: {type(cfg.class_names)}")
    
    # Try to pickle
    try:
        pickle.dumps(cfg)
        print("Original config is picklable.")
    except TypeError as e:
        print(f"Original config pickle failed: {e}")

    # Try to modify
    try:
        cfg.class_names = list(cfg.class_names)
        print(f"Modified class_names type: {type(cfg.class_names)}")
        pickle.dumps(cfg)
        print("Modified config is picklable.")
    except Exception as e:
        print(f"Modification or pickle failed: {e}")

except Exception as e:
    print(f"Error: {e}")
