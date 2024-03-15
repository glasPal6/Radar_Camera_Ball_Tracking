import numpy as np
from scipy.spatial.transform import Rotation as R_rot
from radar_calibration import data_extraction, radar_Kalman, radar_LS 

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

    A_K, R_K, _ = radar_Kalman(targets, x, noise)
    print(A_K)
    print()

    A_LS , R_LS, _ = radar_LS(np.hstack(targets), np.hstack(x))
    print(A_LS)
    print()

    print(np.linalg.norm(A - A_K) ** 2)
    print(np.linalg.norm(A - A_LS) ** 2)
    print()
    print(np.linalg.norm(r_matrix - R_K) ** 2)
    print(np.linalg.norm(r_matrix - R_LS) ** 2)
    print()

def test_radar_calib():
    # Data Paths
    data_path = "Calibration_Data/Test_2024_03_14/pymmw_2024-03-14_13-20-03.log"
    gt_path = "Calibration_Data/Test_2024_03_14/gt_coords.txt"
    config_path = "Calibration_Data/Test_2024_03_14/config_2024-03-14_13-20-06.json"
    reflector_path = "Calibration_Data/Test_2024_03_14/reflector_coords_2024-03-11_10-20-06.txt"

    # Extract data
    gt_positions, max_points = data_extraction(data_path, gt_path, config_path, reflector_path)

    # Estimate the calibration
    A_estimate, R_estimate, t_estimate = radar_LS(np.hstack(gt_positions), np.hstack(max_points))
    # A_estimate, R_estimate, t_estimate = radar_Kalman(gt_positions, max_points, 0.01)
    print(A_estimate)
    print()

    i = 0
    test = A_estimate @ gt_positions[0, :, i]
    print(test)
    print(max_points[0, :, i])
    print(np.linalg.norm(test - max_points[0, :, i]) ** 2)


if __name__ == "__main__":
    # test_algorithms()
    test_radar_calib()

