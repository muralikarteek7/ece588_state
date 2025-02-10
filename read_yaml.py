import yaml
import numpy as np

# Load YAML file
yaml_file = "camera_intrinsics.yaml"  # Update this path if needed

try:
    with open(yaml_file, "r") as f:
        camera_data = yaml.safe_load(f)
    
    # Extract values from YAML
    zed_K = np.array(camera_data["zed_K"])
    zed_intrinsics = camera_data["zed_intrinsics"]
    zed_w = camera_data["zed_w"]
    zed_h = camera_data["zed_h"]

    # Print values to verify
    print("zed_K:", zed_K)
    print("zed_intrinsics:", zed_intrinsics)
    print("zed_w:", zed_w)
    print("zed_h:", zed_h)

except FileNotFoundError:
    print(f"Error: {yaml_file} not found.")
except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")
