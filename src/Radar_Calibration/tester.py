import numpy as np
from numpy.random import beta
from scipy.spatial.transform import Rotation as R_rot
from radar_calibration import data_extraction
from external_calibration import radar_LS

def test_algorithms():
    alpha = 40
    beta = 14
    gamma = 47
    r = R_rot.from_euler("xyz", [alpha, beta, gamma], degrees=True)
    r_matrix = r.as_matrix()
    # print_results(r_matrix, beta, gamma, alpha)

    t = np.array([10, -8, -3.5])
    A = np.hstack([r_matrix, t.reshape((-1, 1))])
    print(A)
    print()

    cols = 5
    times = 120
    targets = np.vstack([np.random.random((3, cols)), np.ones(cols)])
    x = A @ targets
    x = np.array([x] * times)
    targets = np.array([targets] * times)
    noise = 0.01
    x[:, :2, :] = x[:, :2, :] + np.random.normal(0, noise, (times, 2, cols))

    print()

def test_radar_calib():
    # Data Paths
    data_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/pymmw_2024-03-14_13-20-03.log"
    gt_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/ground_truth.txt"
    config_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/config_2024-03-14_13-20-06.json"
    reflector_path = "../../../Calibration_Data/Radar_Calibration/Test_2024_03_14/reflector_coords_2024-03-11_10-20-06.txt"

    # Extract data
    gt_positions, max_points = data_extraction(data_path, gt_path, config_path, reflector_path)

    # Estimate the calibration
    A_estimate, R_estimate, t_estimate = radar_LS(np.hstack(gt_positions), np.hstack(max_points))
    print(A_estimate)
    cos_beta = np.sqrt(R_estimate[0, 0] ** 2 + R_estimate[1, 0] ** 2)
    beta = np.arccos(cos_beta)
    alpha = np.arcsin(R_estimate[2, 1] / cos_beta)
    gamma = np.arccos(R_estimate[0, 0] / cos_beta)
    print(alpha * 180 / np.pi, beta * 180 / np.pi, gamma * 180 / np.pi)
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

