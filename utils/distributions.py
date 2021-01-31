import numpy as np
import tensorflow as tf


def tf_2d_normal(g, x, y, mux, muy, sx, sy, rho):
    '''
    Function that computes a multivariate Gaussian
    Equation taken from 24 & 25 in Graves (2013)
    '''
    with g.as_default():
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)

        sxsy = tf.multiply(sx, sy)

        z = tf.square(tf.divide(normx, sx)) + \
            tf.square(tf.divide(normy, sy)) - \
            2*tf.divide(
                tf.multiply(
                    rho,
                    tf.multiply(normx, normy)),
                sxsy
            )
        negatedRho = 1 - tf.square(rho)

        result = tf.divide(-z, 2*negatedRho)
        denominator = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negatedRho))

        result = tf.divide(tf.exp(result), denominator)

        return result

def tf_1d_normal(g, x, mux, sx):
    with g.as_default():
        denom = 2 * np.pi * sx

        # x_minus_mu  = tf.subtract(x, mux)
        # xmm_vrez_sx = tf.divide(x_minus_mu, sx)
        # na_kvadrat  = tf.square(xmm_vrez_sx)
        # expa        = tf.exp(result)
        # razdel      = tf.divide(expa, denom)

        normx = tf.subtract(x, mux)
        result = -0.5 * tf.square(tf.divide(normx, sx))
        denom = 2 * np.pi * sx
        result = tf.divide(tf.exp(result), denom)
        return result

def tf_1d_lognormal(g, x, mu, sigma):
    with g.as_default():
        norm = tf.subtract(x, mu)
        result = -0.5 * tf.square(tf.divide(norm, sigma))
        denomSqrtTwoPi = np.log(np.sqrt(2.0 * np.pi))
        result = result - denomSqrtTwoPi - tf.log(sigma)
        return result

def tf_bernoulli(g, t, mu):
    '''
    Function that computes a multivariate Gaussian
    Equation taken from 24 & 25 in Graves (2013)
    '''
    with g.as_default():
        edno_minus_p = tf.subtract(1., mu)
        edno_minus_t = tf.subtract(1., t)
        p_po_t = tf.multiply(mu, t)
        emp_po_emt = tf.multiply(edno_minus_p, edno_minus_t)
        sbor = tf.add(p_po_t, emp_po_emt)

        result = tf.add(
            tf.multiply(mu, t),
            tf.multiply(
                tf.subtract(1., mu),
                tf.subtract(1., t)
            )
        )

        return result, edno_minus_p, edno_minus_t, p_po_t, emp_po_emt, sbor

def get_final_error(predicted_traj, true_traj, observed_length, max_num_agents):
    '''
    Function that computes the final euclidean distance error between the
    predicted and the true trajectory
    some data points start from the middle of the batch...doesnt seem right ...
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    taken from: https://github.com/vvanirudh/social-lstm-tf
    '''
    # The data structure to store all errors
    error = 0.0
    # For the last point in the predicted part of the trajectory
    # The predicted position
    pred_pos = predicted_traj[-1, :]
    # The true position
    true_pos = true_traj[-1, :]
    timestep_error = 0
    counter = 0
    for j in range(max_num_agents):
        if true_pos[j, 0] == 0.0:
            continue
        elif pred_pos[j, 0] == 0.0:
            continue
        else:
            if true_pos[j, 1] > 1.0 or true_pos[j, 1] < 0.0:
                continue
            elif true_pos[j, 2] > 1.0 or true_pos[j, 2] < 0.0:
                continue

            # The euclidean distance is the error
            timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]]) # np.sum((true_pos[j, [1, 2]] - pred_pos[j, [1, 2]]) ** 2)
            counter += 1 # the error is divided by the number of agents in the frame ?

    if counter != 0:
        error = timestep_error / counter

    # Return the final error
    return error

def get_mean_error(predicted_traj, true_traj, observed_length, max_num_agents):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    taken from: https://github.com/vvanirudh/social-lstm-tf
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(max_num_agents):
            if true_pos.shape[-1]> 2 and true_pos[j, 0] == 0.0:
                continue
            elif pred_pos.shape[-1]> 2 and pred_pos[j, 0] == 0.0:
                continue
            else:
                if (true_pos.shape[-1]> 2 and true_pos[j, 1] > 1.0) or (true_pos.shape[-1]> 2 and true_pos[j, 1] < 0.0):
                    continue
                elif (true_pos.shape[-1]> 2 and true_pos[j, 2] > 1.0) or (true_pos.shape[-1]> 2 and true_pos[j, 2] < 0.0):
                    continue

                # The euclidean distance is the error
                if true_pos.shape[-1] > 2 :
                    timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]]) # np.sum((true_pos[j, [1, 2]] - pred_pos[j, [1, 2]]) ** 2) 
                elif true_pos.shape[-1] == 2:
                    timestep_error += np.sum((true_pos - pred_pos) ** 2)
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

    # Return the mean error
    return np.mean(error)

# This is too complicated, it takes into account both per_frame and per_agent and it also
# does some weird stuff for differently preprocessed data ...
# def get_mean_error(predicted_traj, true_traj, observed_length, max_num_agents, per_frame=True):
#     '''
#     Function that computes the mean euclidean distance error between the
#     predicted and the true trajectory
#     params:
#     predicted_traj : numpy matrix with the points of the predicted trajectory
#     true_traj : numpy matrix with the points of the true trajectory
#     observed_length : The length of trajectory observed
#     taken from: https://github.com/vvanirudh/social-lstm-tf
#     '''
#     # The data structure to store all errors
#     error = np.zeros(len(true_traj) - observed_length)
#     # For each point in the predicted part of the trajectory
#     for i in range(observed_length, len(true_traj)):
#         # The predicted position
#         pred_pos = predicted_traj[i, :]
#         # The true position
#         true_pos = true_traj[i, :]
#         timestep_error = 0
#         counter = 0
#         for j in range(max_num_agents):
#             if true_pos.shape[-1] > 2 and per_frame and true_pos[j, 0] == 0.0:
#                 continue
#             elif true_pos.shape[-1] > 2 and per_frame and pred_pos[j, 0] == 0.0:
#                 continue
#             elif true_pos.shape[-1] > 2:
#                 if true_pos[j, 1] > 1.0 or per_frame and true_pos[j, 1] < 0.0:
#                     continue
#                 elif true_pos[j, 2] > 1.0 or per_frame and true_pos[j, 2] < 0.0:
#                     continue

#                 # The euclidean distance is the error
#                 timestep_error += np.sum((true_pos[j, [1, 2]] - pred_pos[j, [1, 2]]) ** 2)#np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
#                 counter += 1
#             else:
#                 if true_pos[0] > 1.0 or true_pos[0] < 0.0:
#                     continue
#                 elif true_pos[1] > 1.0 or true_pos[1] < 0.0:
#                     continue

#                 # The euclidean distance is the error
#                 timestep_error += np.sum((true_pos[[0, 1]] - pred_pos[[0, 1]]) ** 2)#np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
#                 counter += 1

#         if counter != 0:
#             error[i - observed_length] = timestep_error / counter

#     # Return the mean error
#     return np.mean(error)

def get_rms_step_error(predicted_map, true_map):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true maps
    '''
    # The data structure to store all errors
    error = np.zeros(true_map.shape)
    for step in range(true_map.shape[0]):
        for idx in range(true_map.shape[1]):
            error[step,idx] += (true_map[step][idx] - predicted_map[step][idx])**2

    # Return the mean error
    return np.sqrt(np.mean(error))

def get_rms_map_error(predicted_map, target_map, total=True):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true maps.
    The predicted map is an average map over all obtained samples.

    Function can compute the difference between the indices that were actually
    supposed to be occupied only, thus excluding the scaling problems.
    '''
    # The data structure to store all errors
    target_map = np.true_divide(np.sum(target_map, axis=0), target_map.shape[0])
    predicted_map = np.true_divide(np.sum(predicted_map, axis=0), predicted_map.shape[0])

    if total:
        indices = [i for i in range(target_map.shape[0])]
    else:
        indices = [i for i, pix in enumerate(target_map) if pix > 0.]

    error = (predicted_map[indices] - target_map[indices])**2
    # Return the root mean squared error
    return np.sqrt(np.mean(error))

def compute_mse(a, b):
    return np.mean((a-b)**2)

def psnr(max_val, mse):
    return 20 * np.log(max_val) / np.log(10.0) - ((10/np.log(10)) * np.log(mse))

def compute_psnr(image, target_image, max_val):
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # PSNR is computed correctly.
    max_val = np.round(max_val).astype(np.float32)
    a = np.round(image).astype(np.float32)
    b = np.round(target_image).astype(np.float32)

    mse = compute_mse(a, b)
    psnr_val = psnr

    return psnr_val(max_val, mse) if mse > 0.0 else 100
def _axis(keep_axis, ndims):
    if keep_axis is None:
        axis = None
    else:
        axis = list(range(ndims))
        try:
            for keep_axis_ in keep_axis:
                axis.remove(keep_axis_)
        except TypeError:
            axis.remove(keep_axis)
        axis = tuple(axis)
    return axis

def structural_similarity_np(true, pred, K1=0.01, K2=0.03, sigma=1.5, win_size=None,
                             data_range=1.0, gaussian_weights=False,
                             use_sample_covariance=True, keep_axis=None):
    from skimage.measure import compare_ssim
    kwargs = dict(K1=K1, K2=K2,
                  win_size=win_size,
                  data_range=data_range,
                  multichannel=True,
                  gaussian_weights=gaussian_weights,
                  sigma=sigma,
                  use_sample_covariance=use_sample_covariance)
    assert true.shape == pred.shape
    shape = true.shape
    true = true.reshape((-1,) + shape[-3:])
    pred = pred.reshape((-1,) + shape[-3:])
    ssim = []
    for true_y, pred_y in zip(true, pred):
        ssim.append(compare_ssim(true_y, pred_y, **kwargs))
    ssim = np.reshape(ssim, shape[:-3])
    return np.mean(ssim, axis=_axis(keep_axis, ssim.ndim))

def errors(outputs, targets):
    # Errors:
    total_err = get_rms_map_error(outputs, targets, total=True)
    occupancy_err = get_rms_map_error(outputs, targets, total=False)
    first_err = get_rms_map_error(outputs[0].reshape((1, outputs[0].shape[0])), targets[0].reshape((1, outputs[0].shape[0])), total=True)
    last_err = get_rms_map_error(outputs[-1].reshape((1, outputs[-1].shape[0])), targets[-1].reshape((1, outputs[-1].shape[0])), total=True)

    print("total_err     : ", total_err)
    print("occupancy_err : ", occupancy_err)
    print("first_err     : ", first_err)
    print("last_err      : ", last_err)
    
    return total_err, occupancy_err, first_err, last_err

def get_pi_idx(x, pdf, rnd):
    # samples from a categorial distribution
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    random_value = rnd.randint(N)
    return random_value

def choose_moments(o_pi, o_mu, o_sx, z_size, temperature, rnd):
    logmix2 = np.copy(o_pi)/temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(z_size, 1)

    mixture_idx = np.zeros(z_size)
    chosen_mean = np.zeros(z_size)
    chosen_logstd = np.zeros(z_size)
    for j in range(z_size):
        idx = get_pi_idx(rnd.rand(), logmix2[j], rnd)
        mixture_idx[j] = idx
        chosen_mean[j] = o_mu[j][idx]
        chosen_logstd[j] = o_sx[j][idx]

    return chosen_mean, chosen_logstd

def sample_2d_normal(mux, muy, sx, sy, rho, rnd):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # Extract mean
    mean = np.array([mux, muy]).reshape((2))
    cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]).reshape((2,2))
    # Sample a point from the multivariate normal distribution
    x = rnd.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def sample_bernoulli(o_mu):
    '''
    Function that samples from a multivariate Gaussian
    That has the statistics computed by the network.
    '''
    from scipy.stats import bernoulli
    
    sample = np.zeros(len(o_mu))
    for i, p in enumerate(o_mu):
        sample[i] = bernoulli.rvs(p=p[0])

    return sample
