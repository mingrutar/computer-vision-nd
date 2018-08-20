from robot_class import robot
from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------
# this helper function displays the world that a robot is in
# it assumes the world is a square grid of some given size
# and that landmarks is a list of landmark positions(an optional argument)
def display_world(world_size, position, landmarks=None):
    
    # using seaborn, set background grid to gray
    sns.set_style("dark")

    # Plot grid of values
    world_grid = np.zeros((world_size+1, world_size+1))

    # Set minor axes in between the labels
    ax=plt.gca()
    cols = world_size+1
    rows = world_size+1

    ax.set_xticks([x for x in range(1,cols)],minor=True )
    ax.set_yticks([y for y in range(1,rows)],minor=True)
    
    # Plot grid on minor axes in gray (width = 1)
    plt.grid(which='minor',ls='-',lw=1, color='white')
    
    # Plot grid on major axes in larger width
    plt.grid(which='major',ls='-',lw=2, color='white')
    
    # Create an 'o' character that represents the robot
    # ha = horizontal alignment, va = vertical
    ax.text(position[0], position[1], 'o', ha='center', va='center', color='r', fontsize=30)
    
    # Draw landmarks if they exists
    if(landmarks is not None):
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in landmarks:
            if(pos != position):
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)
    
    # Display final result
    plt.show()

def display_world_2(world_size, positions, landmarks=None):
    # using seaborn, set background grid to gray
    sns.set_style("dark")

    # Plot grid of values
    world_grid = np.zeros((world_size + 1, world_size + 1))

    # Set minor axes in between the labels
    ax = plt.gca()
    cols = world_size + 1
    rows = world_size + 1

    ax.set_xticks([x for x in range(1, cols)], minor=True)
    ax.set_yticks([y for y in range(1, rows)], minor=True)

    # Plot grid on minor axes in gray (width = 1)
    plt.grid(which='minor', ls='-', lw=1, color='white')

    # Plot grid on major axes in larger width
    plt.grid(which='major', ls='-', lw=2, color='white')

    # Create an 'o' character that represents the robot
    # ha = horizontal alignment, va = vertical
    for i, pos in enumerate(positions):
        if pos:
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='r', fontsize=30)

    # Draw landmarks if they exists
    if (landmarks is not None):
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in landmarks:
            if pos not in positions:
                print('display_world_2, lm=', pos[0], pos[1])
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)

    # Display final result
    plt.show()

def display_world_3(world_size, slam_pos, slam_landmarks, rob_pos, rob_landmarks):
    # slam and expect are [last_pos, landmark]
    # using seaborn, set background grid to gray
    sns.set_style("dark")
    # Plot grid of values
    world_grid = np.zeros((world_size + 1, world_size + 1))

    # Set minor axes in between the labels
    ax = plt.gca()
    cols = world_size + 1
    rows = world_size + 1

    ax.set_xticks([x for x in range(1, cols)], minor=True)
    ax.set_yticks([y for y in range(1, rows)], minor=True)

    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color(None)
    ax.patch.set_facecolor('0.1')
    
    # Plot grid on minor axes in gray (width = 1)
    plt.grid(which='minor', ls='-', lw=1, color='white')

    # Plot grid on major axes in larger width
    plt.grid(which='major', ls='-', lw=2, color='white')

    # Create an 'o' character that represents the robot
    # ha = horizontal alignment, va = vertical
    slam_X = [slam_pos[0]]
    ax.scatter(slam_pos)
    plt.plot(X,Y)
    ax.text(slam_pos[0], slam_pos[1], 'S', ha='center', va='center', color='r', fontsize=30)
    ax.text(rob_pos[0], rob_pos[1], 'R', ha='center', va='center', color='g', fontsize=30)
    # Draw landmarks if they exists
    if slam_landmarks:
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in slam_landmarks:
            if pos not in slam_pos:
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)
    if rob_landmarks:
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in rob_landmarks:
            if pos not in slam_pos:
                ax.text(pos[0], pos[1], 'o', ha='center', va='center', color='b', fontsize=20)

    # Display final result
    plt.show()

def get_data_by_x_y(coor_list):
    ''' make X, Y from list of [x,y] '''
    X, Y = [], []
    for v in coor_list:
        X.append(v[0])
        Y.append(v[1])
    return X, Y

def get_data_x_y(inputs):
    ''' make_data generates (dx, dy) for movement. convert the dx, dy to x, y. Then make X, Y
    '''
    coords = [[world_size / 2.0, world_size / 2.0]]
    for i, dpos in enumerate(inputs):
        coords.append([dpos[1][0] + coords[i][0], dpos[1][1] + coords[i][1]])
    return get_data_by_x_y(coords)

def plot_all(data_landmarks, landmarks, data_poses, poses, noise):
    ''' plot for single comparison '''
    # landmarks from data
    LX_i, LY_i = get_data_by_x_y(data_landmarks)
    # estimated landmarks 
    LX_d, LY_d = get_data_by_x_y(landmarks)
    print('real landmarks:', LX_i, LY_i, 'estimated landmarks:', LX_d, LY_d)
    # move poses from data
    CX_i, CY_i = get_data_by_x_y(data_poses)
    # estimated move poses
    CX_d, CY_d = get_data_by_x_y(poses)
    n = 4
#     print('poses of data:', CX_i[:n], CY_i[:n])
#     print('estimated poses:', CX_d[:n], CY_d[:n])
    ax = plt.gca()
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.title("Data vs Estimate, noises = " + str(noise))
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.scatter(LX_i, LY_i, color='red', marker='o', alpha=1.0, label='Landmarks from Data', s=80)
    plt.scatter(LX_d, LY_d, color='green', marker='x', alpha=0.8, label='Landmarks Estimated', s=100)
    plt.plot(CX_i, CY_i, color='blue', marker='o', markerfacecolor='blue', markersize=12, label='Path from Data')
    plt.plot(CX_d, CY_d, color='orange', linestyle='dashed', marker='o', markerfacecolor='orange', markersize=12, \
             label='Path Estimated')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fancybox=True, shadow=True, ncol=2, frameon=True)
    plt.show()
# --------
# this routine makes the robot data
# the data is a list of measurements and movements: [measurements, [dx, dy]]
# collected over a specified number of time steps, N
#
def make_data(N, num_landmarks, world_size, measurement_range, motion_noise, 
              measurement_noise, distance):


    # check if data has been made
    complete = False

    while not complete:
        data = []

        # make robot and landmarks
        r = robot(world_size, measurement_range, motion_noise, measurement_noise)
        r.make_landmarks(num_landmarks)
        seen = [False for row in range(num_landmarks)]
    
        # guess an initial motion
        orientation = random.random() * 2.0 * pi
# for test        dx = round(cos(orientation) * distance)
#         dy = round(sin(orientation) * distance)
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance
        
        for k in range(N-1):
    
            # collect sensor measurements in a list, Z
            Z = r.sense()

            # check off all landmarks that were observed 
            for i in range(len(Z)):
                seen[Z[i][0]] = True
    
#             print('make_data: robot.x,y=({},{}), dx,dy=({},{}), Z={}'.format(r.x, r.y, dx, dy, Z))
            # move
            while not r.move(dx, dy):
                # if we'd be leaving the robot world, pick instead a new direction
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            # collect/memorize all sensor and motion data
            data.append([Z, [dx, dy]])

        # we are done when all landmarks were observed; otherwise re-run
        complete = (sum(seen) == num_landmarks)
#         print('make_data: complete', complete)
        # add landmark measurements for the last displacement
#         if complete:
#             Z = r.sense()
#             data.append([Z, []])

    print(' ')
    print('Landmarks: ', r.landmarks)
    print(r)

    return data, r