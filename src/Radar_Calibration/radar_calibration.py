import numpy as np

def radar_LS(targets, measurements):
    A = targets @ measurements.T @ np.linalg.inv(measurements @ measurements.T)
    R = A[:2, :2]
    t = A[:2, 2]
    return A, R, t

def radar_Kalman(targets, measurements, noise_m, noise_p=0.001):
    """
        targets = (no. of images in time, 3, no. of reflectors)
        measurements = (no. of images in time, 3, no. of reflectors)
        noise_n = ()
    """
    # Initial estimate of the values
    _, R, t = radar_LS(targets[0, :, :], measurements[0, :, :])
    mu = np.array([
        t[0], t[1],
        np.mean([
            np.arccos(R[0, 0]),
            np.arccos(R[1, 1]),
            np.arcsin(R[0, 1]),
            -1*np.arcsin(R[1, 0]),
        ])
    ])
    sigma = noise_p * np.eye(3)

    for i in range(1, targets.shape[0]):
        # Prediction step
        # mu = mu
        sigma = sigma + noise_p * np.eye(3)

        # Update step
        Rs = np.array([
            [-np.sin(mu[2]), np.cos(mu[2])],
            [-np.cos(mu[2]), -np.sin(mu[2])],
        ]) @ measurements[i, :2, :]
        Hx = np.vstack([np.eye(2) for _ in range(Rs.shape[1])])
        Rs = Rs.flatten('F')
        Rs = Rs.reshape((-1, 1))
        Hx = np.hstack([Hx, Rs])

        H = np.array([
            [np.cos(mu[2]), np.sin(mu[2]), mu[0]],
            [-np.sin(mu[2]), np.cos(mu[2]), mu[1]],
            [0, 0, 1],
        ])

        K = sigma @ Hx.T @ np.linalg.inv(Hx @ sigma @ Hx.T + noise_m * np.eye(2*(measurements.shape[2])))
        sigma = sigma - K @ Hx @ sigma
        mu = mu + (K @ ((targets[i, :, :] - H @ measurements[i, :, :])[:2, :].flatten('F').reshape((-1, 1)))).flatten()

    A = np.array([
        [np.cos(mu[2]), np.sin(mu[2]), mu[0]],
        [-np.sin(mu[2]), np.cos(mu[2]), mu[1]],
        [0, 0, 1],
    ])
    R = A[:2, :2]
    t = A[:2, 2]
    return A, R, t

