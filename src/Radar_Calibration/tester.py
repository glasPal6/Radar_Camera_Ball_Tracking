import numpy as np
from radar_calibration import data_extraction
from external_calibration import trilateration, rotation_least_squares

def test_algorithms():
    from scipy.spatial.transform import Rotation as R_rot
    np.random.seed(0)

    alpha = 50
    beta = 39
    gamma = 67
    r = R_rot.from_euler("xyz", [alpha, beta, gamma], degrees=True)
    r_matrix = r.as_matrix()
    # r_matrix = np.eye(3)

    t = np.array([10, -8, -3.5])
    design_matrix = np.hstack([r_matrix, t.reshape((-1, 1))])

    cols = 5
    times = 2
    targets = np.vstack([np.random.random((3, cols)), np.ones(cols)])
    x = design_matrix @ targets

    range_measurements = np.linalg.norm(x, axis=0)
    azimuth_measurements = np.arctan2(x[1, :], x[0, :])
    measurements = np.vstack([
        range_measurements * np.cos(azimuth_measurements), 
        range_measurements * np.sin(azimuth_measurements),
    ])

    x = np.array([x] * times)
    targets = np.array([targets] * times)
    measurements = np.array([measurements] * times)

    c_r = trilateration(np.hstack(targets), np.hstack(measurements))
    R = rotation_least_squares(np.hstack(targets), np.hstack(measurements), c_r)
    print(r_matrix)
    print()
    print(R)
    print()
    print()
    print(t)
    print(-R @ c_r.T)

def test_radar_calib():
    # Data Paths
    data_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/pymmw_2024-03-14_13-20-03.log"
    gt_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/ground_truth.txt"
    config_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/config_2024-03-14_13-20-06.json"
    reflector_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/reflector_coords_2024-03-11_10-20-06.txt"

    # Extract data
    gt_positions, max_points = data_extraction(data_path, gt_path, config_path, reflector_path)

    # Estimate the calibration
    radar_centre = trilateration(np.hstack(gt_positions), np.hstack(max_points))
    rot_mat = rotation_least_squares(np.hstack(gt_positions), np.hstack(max_points), radar_centre)
    # u, _, vt = np.linalg.svd(rot_mat)
    # rot_mat = u @ vt
    A_estimate = np.hstack([rot_mat, -rot_mat @ radar_centre.T])
    print(-rot_mat @ radar_centre.T)
    print()
    print(rot_mat)
    print(rot_mat @ rot_mat.T)
    print()

    for i in range(gt_positions.shape[2]):
        test = A_estimate @ gt_positions[0, :, i]
        test = test[:2]
        print(max_points[0, :, i])
        print(test)
        print(np.linalg.norm(test - max_points[0, :2, i]) ** 2)
        print()

if __name__ == "__main__":
    # test_algorithms()
    test_radar_calib()

