import numpy as np
from radar_calibration import radar_Kalman, radar_LS 

if __name__ == "__main__":
    theta = 1
    t = np.array([10, -8])
    A = np.array([
        [np.cos(theta), np.sin(theta), t[0]],
        [-1*np.sin(theta), np.cos(theta), t[1]],
        [0, 0, 1],
    ])
    print(A)
    print()

    cols = 10
    times = 10
    x = np.vstack([
        np.random.random((2, cols)),
        np.ones(cols)
    ])
    targets = A @ x
    x = np.array([x]*times)
    targets = np.array([targets]*times)
    noise = 0.1
    x[:, :2, :] = x[:, :2, :] + np.random.normal(0, noise, (times, 2, cols))

    estimate_A_K, _, _ = radar_Kalman(targets, x, noise)
    print(estimate_A_K)
    print()

    estimate_A_LS, _, _ = radar_LS(np.hstack(targets), np.hstack(x))
    print(estimate_A_LS)
    print()

    print(np.linalg.norm(A - estimate_A_K)**2)
    print(np.linalg.norm(A - estimate_A_LS)**2)
