import numpy as np
import json
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import scipy.interpolate as spi

def extract_azimuth_data(config, azimuth_data):
    # Extract config
    tx_azimuth_antennas = config['Azimuth antennas']
    rx_antennas = config['Receive antennas']
    range_bins = config['Range bins'] 
    angle_bins = config['Doppler bins']
    range_res = config['Range resolution (m)'] 
    range_bias = config['Range Bias (m)'] if "Range Bias (m)" in config.keys() else 0
    
    # define the space
    t = np.array(range(-angle_bins//2 + 1, angle_bins//2)) * (2 / angle_bins)
    t = np.arcsin(t)
    r = np.array(range(range_bins)) * range_res

    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400

    xi = np.linspace(-range_width, range_width, grid_res)
    yi = np.linspace(0, range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)

    x = np.array([r]).T * np.sin(t)
    y = np.array([r]).T * np.cos(t)
    y = y - range_bias

    a = np.copy(azimuth_data)
    a = np.array([a[i] + 1j * a[i+1] for i in range(0, len(a), 2)])
    a = np.reshape(a, (range_bins, tx_azimuth_antennas * rx_antennas))
    a = np.fft.fft(a, angle_bins)
    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right       
    a = a[:,1:]  # cut off first angle bin
      
    # Interpolate the data
    zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), (xi, yi), method='linear')
    zi = zi[:-1,:-1]

    return zi[::-1, ::-1]

def plot_calibration_image(config, azimuth_data):
    # Extract config
    range_bins = config['Range bins'] 
    angle_bins = config['Doppler bins']
    range_res = config['Range resolution (m)'] 
    
    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400 -1

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)  # rows, cols, idx
    fig.tight_layout()
    # cm = ax.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, alpha=0.95)
    cm = ax.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)
                       
    ax.set_title('Azimuth-Range FFT Heatmap [{};{}]'.format(angle_bins, range_bins), fontsize=10)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')
    
    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)    

    ax.set_ylim([0, +range_depth])
    ax.set_xlim([-range_width, +range_width])

    # draw 45 degree lines
    for i in range(1, int(range_depth)+1):
        ax.add_patch(pat.Arc((0, 0), width=i*2, height=i*2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))

    # Plot the data
    data = extract_azimuth_data(config, azimuth_data)
    cm.set_array(data)  # rotate 180 degrees
    cm.autoscale()

    return fig, ax, cm

def get_radar_points(config, azimuth_data, reflector_coordinates_path):
    """
    Get the corner reflector points
    """
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
        ax.add_patch(rect)

        # Write the data to a file to be read later
        with open(reflector_coordinates_path, 'a') as f:
            f.write(f"{(x1+4)*400/8 - 1}, {400 - 1 - y1*400/8}\n")
            f.write(f"{(x2+4)*400/8 - 1}, {400 - 1 - y2*400/8}\n")
            
    
    # Clear the file
    with open(reflector_coordinates_path, 'w') as f:
        f.write("")

    # Extract config
    range_bins = config['Range bins'] 
    range_res = config['Range resolution (m)'] 
    
    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400

    fig, ax, cm = plot_calibration_image(config, azimuth_data)
    rs = RectangleSelector(ax, line_select_callback,
                       useblit=True,
                       button=[1, 3],  # don't use middle button
                       minspanx=5, minspany=5,
                       spancoords='pixels',
                       interactive=True)
    plt.show()

def get_maximun_points(config, azimuth_data, reflector_coordinates_path):
    """
    Get the maximum points
    """
    # Extract config
    range_bins = config['Range bins'] 
    range_res = config['Range resolution (m)'] 
    
    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400

    # Extract and group the points
    cnr_ref = np.loadtxt(reflector_coordinates_path, delimiter=",").reshape((-1, 2, 2))
    data = extract_azimuth_data(config, azimuth_data)

    # Find the maximuns
    max_points = []
    for sqr in cnr_ref:
        x1, x2 = int(np.round(sqr[0, 0])), int(np.round(sqr[1, 0]))
        y1, y2 = int(np.round(sqr[0, 1])), int(np.round(sqr[1, 1]))
        max = np.unravel_index(np.argmax(data[y2:y1, x1:x2]), data[y2:y1, x1:x2].shape)
        x_point = (max[1]+x1 + 1) * range_depth / grid_res - range_width
        y_point = (grid_res - 1 - (max[0]+y2)) * range_depth / grid_res
        # Note that the points are swapped around
        max_points.append([y_point, x_point])

    return np.array(max_points).T

def data_extraction(data_path, gt_positions_path, config_path, reflector_coordinates_path):
    """
    This assumes satic positions of the corner reflectors
    Variables:
        - path: path to the file
        - gt_positions - path to the ground truth positions of the known objects
    """
    # Load the files
    data = []
    with open(data_path, 'r') as f:
        data_raw = f.read()
        data_raw = data_raw.replace("'", '"')
        data_raw = data_raw.splitlines()
        for d in data_raw:
            data.append(json.loads(d))
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Write the calibration points to a file
    # get_radar_points(config, np.array(data[0]['dataFrame']['azimuth_static']), reflector_coordinates_path)

    # Get the max points for each time step
    max_points = []
    for i in range(len(data)):
        print(f"{i} / {len(data)}", end="\r")
        max_points_i = get_maximun_points(config, np.array(data[i]['dataFrame']['azimuth_static']), reflector_coordinates_path)
        # max_points_i = np.vstack([max_points_i, np.ones((1, max_points_i.shape[1]))])
        max_points.append(max_points_i)
    max_points = np.array(max_points)

    # Load the ground truth positions
    gt_positions = []
    with open(gt_positions_path, 'r') as f:
        data_raw = f.readlines()
        for d in data_raw:
            gt_positions.append(np.array([float(i) for i in d.split(",")]))
    gt_positions = np.array(gt_positions).T
    gt_positions = np.vstack([gt_positions, np.ones((1, gt_positions.shape[1]))])
    gt_positions = np.array([gt_positions]*len(data))

    return gt_positions, max_points

