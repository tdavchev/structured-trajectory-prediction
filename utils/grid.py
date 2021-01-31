'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np


def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''

    # Maximum number of pedestrians
    mnp = frame.shape[0]
    height, width = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mnp, mnp, grid_size**2))

    width_bound, height_bound = neighborhood_size/(width*1.0), neighborhood_size/(height*1.0)

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # If pedID is zero, then non-existent ped
        if frame[pedindex, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue

        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

        height_low, height_high = current_x - height_bound/2, current_x + height_bound/2
        width_low, width_high = current_y - width_bound/2, current_y + width_bound/2

        # #### visualise
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig,ax = plt.subplots(1)
        # idx = 1
        # ax.plot(frame[idx,2], frame[idx,1], 'x', label=str(idx))
        # # make sure its widht, height and not reversed (it was 0.05 and 0.04 for 576, 720)
        # rect = patches.Rectangle((height_low,width_low), width_bound, height_bound,linewidth=0.6,edgecolor='r',facecolor='none')
        # ax.add_patch(rect)
        # plt.show()
        # #### end visualisation

        # For all the other peds
        for otherpedindex in range(mnp):
            # If other pedID is zero, then non-existent ped
            if frame[otherpedindex, 0] == 0:
                # Binary mask should be zero
                continue

            # If the other pedID is the same as current pedID
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
            if other_y >= width_high or other_y < width_low or other_x >= height_high or other_x < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - height_low)/height_bound) * grid_size))
            cell_y = int(np.floor(((other_y - width_low)/width_bound) * grid_size))

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask


def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    sl = sequence.shape[0]
    mnp = sequence.shape[1]
    sequence_mask = np.zeros((sl, mnp, mnp, grid_size**2))

    for i in range(sl):
        sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)

    return sequence_mask
