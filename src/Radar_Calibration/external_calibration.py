import numpy as np

def radar_LS(targets, measurements):
    """
    targets = (4, no. of reflectors)
    measurements = (3, no. of reflectors)
    noise_n = ()
    """
    A = measurements @ targets.T @ np.linalg.inv(targets @ targets.T)
    R = A[:2, :3]
    t = A[:2, 3]
    return A, R, t

if __name__ == "__main__":
    np.random.seed(0)

    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    cols = 5
    times = 1
    targets = np.vstack([np.random.random((3, cols)), np.ones(cols)])
    x = A @ targets
    x = np.array([x] * times)
    targets = np.array([targets] * times)
    # noise = 0.01
    # x[:, :2, :] = x[:, :2, :] + np.random.normal(0, noise, (times, 2, cols))

    A_esti, _, _ = radar_LS(np.hstack(targets), np.hstack(x))
    print(A_esti)
