import numpy as np

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
    design_matrix = np.vstack([
        np.hstack([
            [x_diff[:, i].T * measurements[1, i] for i in range(x_diff.shape[1])],
            [x_diff[:, i].T * -measurements[0, i] for i in range(x_diff.shape[1])],
        ])
    ])
    design_matrix = design_matrix.T @ design_matrix
    w, v = np.linalg.eig(design_matrix)
    m = np.argmin(w)

    # Required constant so that the properties of R are upheld
    r_row = v[:, m] * -np.sqrt(2)
    rot_matrix = np.vstack([
        r_row[:3],
        r_row[3:],
        np.cross(r_row[:3], r_row[3:]),
    ])
    return rot_matrix

if __name__ == "__main__":
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
    
