import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats

# demo of Bayesian linear regression based on PRML 3.3.1
# reproducing parts of Figure 3.7

D = 2
noise_std = 0.2  # std of the noise in the data, assumed known; see p 154
alpha = 2  # set the variance hyperparam of prior, see eq (3.52)
beta = 25

m0 = np.array([[0.], [0.]])
S0 = 1/alpha*np.eye(D)
S0_inv = np.linalg.inv(S0)


def Generate_New_Observation():
    x = np.random.uniform(-1, 1)
    a0 = -0.3
    a1 = 0.5
    y = np.random.normal(loc = a0 + a1*x, scale = noise_std)
    new_pt = np.array([x, y]).reshape(1, 2)
    return new_pt

# likelihood in simple linear regression as a function of w0, w1
# (x,t) is a pair of data observation
def like(w0, w1, x, t):
    return sp.stats.norm.pdf(t, loc=(w0 + w1 * x), scale=noise_std)

# Draw the lines which has w0 and w1 as bias and slope generated from multivariate normal gaussian
def Draw_Sample_Lines(ax, m, S, n):
    z = np.random.multivariate_normal(mean=m.reshape(D,), cov=S, size=n)
    
    for i in range(n):
        w0 = z[i][0]
        w1 = z[i][1]
        x = np.arange(-1, 1, 0.02)
        y = w0 + w1*x
        ax.plot(x, y, 'r-')

# Calculate New Mean and Covariance Matrices, reference Bishop p153 (Korean Ver p173)
def Calc_New_Mean_Cov(data_list):
    # data_list :
    #  [x0, y0], [x1, y1], ...
    
    # PHI shape N * M
    # [1, X1], [1, X2], ...
    number_pts = data_list.shape[0]
    #print('number_pts', number_pts)
    ones = np.ones(number_pts).reshape(1,number_pts)
    PHI = np.transpose(np.concatenate((ones, data_list.T[0].reshape(1,number_pts)), axis=0))
    
    S1_inv = S0_inv + beta * np.dot(PHI.T, PHI)
    S1 = np.linalg.inv(S1_inv)
    
    # t shape N*1
    # [y0], [y1], ...
    t =  np.transpose(data_list.T[1].reshape(1,number_pts))
    m1 = np.dot(S1 ,(np.dot(S0_inv,m0) + beta*np.dot(PHI.T, t)))
    m1 = m1.reshape(2,)
    return m1, S1

grid = np.arange(-1, 1, 0.02)
xx, yy = np.meshgrid(grid, grid)
np.random.seed(19)

row_idx = 1
tot_row_cnt = 21
tot_col_cnt = 3
data_sampled = np.empty(shape=[0, 2])
n_lines = 8

fig = plt.figure(figsize=(8,70))

# Initialize the mean and covariance
m = m0
S = S0

for row_idx in range(1, tot_row_cnt+1):
    if row_idx != 1:
        # likelihood function for the new observation
        new_data_pt = Generate_New_Observation()
        x1 = new_data_pt[0][0]
        t1 = new_data_pt[0][1]
        data_sampled = np.concatenate((data_sampled, new_data_pt), axis = 0)

        zs = np.array([like(x, y, x1, t1) for x, y in zip(np.ravel(xx), np.ravel(yy))])
        zz_l = zs.reshape(xx.shape)
        ax = fig.add_subplot(tot_row_cnt, tot_col_cnt, 3*(row_idx-1)+1)
        ax.pcolor(xx, yy, zz_l)

        if row_idx == 2:
            ax.plot(-0.4, 0.4, 'w-+')

        # calculate new mean and covariance
        m, S = Calc_New_Mean_Cov(data_sampled)

        # Draw new Posterior which has calculated after new likelihood and previouse prior
        zz = np.array([sp.stats.multivariate_normal.pdf([x, y], m, S) for x, y in
                       zip(np.ravel(xx), np.ravel(yy))])
        zz = zz.reshape(xx.shape)

        ax = fig.add_subplot(tot_row_cnt, tot_col_cnt, 3*(row_idx-1)+2)
        ax.pcolor(xx, yy, zz)

        if row_idx == 2:
            ax.plot(-0.4, 0.4, 'w-+')

    else:
        # Draw a isotropic Gaussian Prior before observing any data
        zs = np.array([sp.stats.multivariate_normal.pdf([x, y], m0.reshape(2,), S0) for x, y in
                       zip(np.ravel(xx), np.ravel(yy))])
        zz_p = zs.reshape(xx.shape)

        ax = fig.add_subplot(tot_row_cnt, tot_col_cnt, 3*(row_idx-1)+2)
        ax.pcolor(xx, yy, zz_p)

    # Draw sample lines correspoding observed data
    ax = fig.add_subplot(tot_row_cnt, tot_col_cnt, 3*(row_idx-1)+3)
    
    Draw_Sample_Lines(ax, m, S, n_lines)

    if data_sampled.shape[0] >= 1:
        ax.plot(data_sampled[:, 0], data_sampled[:, 1], 'bo', fillstyle='none')

plt.show()