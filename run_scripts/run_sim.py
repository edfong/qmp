import jax
import jax.numpy as jnp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from tqdm import tqdm_notebook
from qmp.qmp_functions import fit_Q, fit_Q_perm, rearrange_Q_B
from qmp.sample_qmp_functions import PR_loop_B, approx_PR_B
from qmp.utils.bivariate_copula import norm_logbicop_approx

#grid-based
du = 0.005
u_plot = jnp.arange(du, 1, du)
n_plot = len(u_plot)


#Cubic example
Q_truth = 4*(u_plot-0.4)**3 + 0.2*(u_plot) #Try right shifting so uniform is not mean 0
q_truth = 12*(u_plot-0.4)**2 + 0.2


def sim(n,Q_truth,u_plot,seed):
    np.random.seed(seed)
    u = np.random.rand(n)
    y = np.interp(u, u_plot,Q_truth)
    return(y)

#Simulate data
y_full = sim(500,Q_truth,u_plot,seed = 2024) #sim full set - subset so datasets are nested (to decrease variance)

for n in np.array([50,500]):
    print('n = {}'.format(n))
    y = y_full[0:n]

    range_y = np.max(y) - np.min(y)

    # Fit hyperparameters ###
    a = np.sqrt(12)*np.std(y)
    k = 0.5

    c_vals = np.arange(0.05,1.0,0.05)
    n_c = len(c_vals)
    preq_score = np.zeros(n_c)
    n_rearr = np.zeros(n_c)

    start = time.time()
    for i in range(n_c): #parallelize this?
        #Fit
        c = c_vals[i]
        Q_plot,preq_score_vec,n_rearr[i] = fit_Q_perm(np.array([a,c,0.5]),y)
        preq_score[i] = np.mean(preq_score_vec)

    end = time.time()
    print("Estimating c took {} seconds".format(round(end-start,2)))


    #plt.figure()
    c_preq = c_vals[np.argmax(preq_score)]
    print(c_preq)
    c_opt = c_preq

    ## Estimation 
    start = time.time()
    Q_plot,*_ = fit_Q_perm(np.array([a,c_opt,k]),y)
    Q_plot = Q_plot.block_until_ready()
    end = time.time()
    print("Initial estimation took {} seconds".format(round(end-start,2)))

    ## Predictive resampling
    seed = 1706
    B = 5000
    key = jax.random.key(seed)
    key = jax.random.split(key, B)
    T = 5000

    #Exact sampling
    start = time.time()
    Q_pr = PR_loop_B(key, Q_plot,a,c_opt,k,n,T)
    Q_pr_rearr =  rearrange_Q_B(Q_pr)
    end = time.time()
    print("Exact sampling samples took {} seconds".format(round(end-start,2)))

    #GP sampling
    start = time.time()
    Q_pr_gp = approx_PR_B(seed,Q_plot,a,c_opt,k,n,B)
    Q_pr_gp_rearr = rearrange_Q_B(Q_pr_gp)
    end = time.time()
    print("Approx sampling samples took {} seconds".format(round(end-start,2)))

    #Save data
    np.save("../plots/plot_data/Q_pr_rearr_n{}".format(n),Q_pr_rearr)
    np.save("../plots/plot_data/Q_pr_gp_rearr_n{}".format(n),Q_pr_gp_rearr)

    #Save hyperparameters
    hyperparam_opt = {'a' : a, 'c': c_opt, 'k': k,  'T':T, 'B': B}
    np.save("../plots/plot_data/hyperparam_opt_n{}".format(n),hyperparam_opt)

    #Save truth and data
    np.save("../plots/plot_data/y_n{}".format(n),y)
    np.save("../plots/plot_data/Q_truth",Q_truth)
    np.save("../plots/plot_data/u_plot",u_plot)