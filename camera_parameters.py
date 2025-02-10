#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from sensor_msgs.msg import CameraInfo

def camera_info_callback(msg):
    # Extract Intrinsic Camera Matrix (K)
    zed_K = np.array(msg.K).reshape((3, 3)).tolist()  # Convert NumPy array to list
    zed_intrinsics = [float(msg.K[0]), float(msg.K[4]), float(msg.K[2]), float(msg.K[5])]

    # Extract Image Dimensions
    zed_w = int(msg.width)
    zed_h = int(msg.height)

    # Prepare data dictionary
    camera_data = {
        "zed_K": zed_K,
        "zed_intrinsics": zed_intrinsics,
        "zed_w": zed_w,
        "zed_h": zed_h
    }

    # Write data to YAML file
    yaml_file = "camera_intrinsics.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(camera_data, f, default_flow_style=False)

    json_file = "camera_intrinsics.json"
    with open(json_file, "w") as f:
        json.dump(camera_data, f, indent=4)


    rospy.loginfo(f"Camera intrinsics written to {yaml_file}")
    rospy.loginfo(f"Camera intrinsics written to {json_file}")
    rospy.signal_shutdown("Data received")


def main():
    rospy.init_node("camera_intrinsics_saver", anonymous=True)
    rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo, camera_info_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
