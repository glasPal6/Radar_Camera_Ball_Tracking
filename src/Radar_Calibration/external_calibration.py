import numpy as np

def radar_LS(targets, measurements):
    """
    targets = (4, no. of reflectors)
    measurements = (3, no. of reflectors)
    noise_n = ()
    """
    A = measurements @ targets.T @ np.linalg.inv(targets @ targets.T)
    R = A[:3, :3]
    t = A[:3, 3]
    u, s, vt = np.linalg.svd(R)
    R = u @ vt
    return A, R, t

def trilateration(targets, measurements, variances=None):
    """
    targets = (4, no. of reflectors)
    measurements = (3, no. of reflectors)
    noise_n = ()
    """
    targets = targets.T
    measurements = measurements.T

    # Get the range of the measurements
    range_measurements = np.linalg.norm(measurements, axis=1)

    # Construct the matricies
    A = np.hstack([
        np.ones((targets.shape[0], 1)),
        -2 * targets[:, :3],
    ])
    b = np.vstack([
        range_measurements ** 2 - np.sum(targets[:, :3] ** 2, axis=1)
    ]).T

    # x = np.linalg.lstsq(A, b, rcond=None)
    x = np.linalg.inv(A.T @ A) @ A.T @ b

    return x[1:].T

def rotation_least_squares(targets, measurements, radar_centre):
    """
    targets = (4, no. of reflectors)
    measurements = (3, no. of reflectors)
    noise_n = ()
    """
    x_diff = targets[:3, :] - radar_centre.T
    # design_matrix = np.hstack([
    #     x_diff.T * measurements[], x_diff.T
    # ])
    print(design_matrix[0])

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R_rot
    np.random.seed(0)

    alpha = 40
    beta = 14
    gamma = 47
    r = R_rot.from_euler("xyz", [alpha, beta, gamma], degrees=True)
    r_matrix = r.as_matrix()
    # r_matrix = np.eye(3)

    t = np.array([10, -8, -3.5])
    A = np.hstack([r_matrix, t.reshape((-1, 1))])

    cols = 5
    times = 1
    targets = np.vstack([np.random.random((3, cols)), np.ones(cols)])
    x = A @ targets

    range_measurements = np.linalg.norm(x, axis=0)
    azimuth_measurements = np.arctan2(x[1, :], x[0, :])
    mearsurements = np.vstack([
        range_measurements * np.cos(azimuth_measurements), 
        range_measurements * np.sin(azimuth_measurements),
    ])

    x = np.array([x] * times)
    targets = np.array([targets] * times)
    mearsurements = np.array([mearsurements] * times)

    c_r = trilateration(np.hstack(targets), np.hstack(mearsurements))
    R = rotation_least_squares(np.hstack(targets), np.hstack(mearsurements), c_r)
    
