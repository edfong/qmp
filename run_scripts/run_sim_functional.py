import jax
import jax.numpy as jnp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from tqdm import tqdm_notebook
from quantile_MP.qmp_functions import fit_Q, fit_Q_perm, rearrange_Q_B
from quantile_MP.sample_qmp_functions import PR_loop_B, approx_PR_B
from quantile_MP.utils.bivariate_copula import norm_logbicop_approx
#from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde


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
    # Denity
    #a = jnp.ones(n_plot)
    kde_fit = gaussian_kde(y)
    a = 1/kde_fit(np.quantile(y,u_plot))

    # Bandwidth
    k = 0.5
    c_opt = 0.7

    ## Estimation 
    start = time.time()
    Q_plot,*_ = fit_Q_perm([a,c_opt,k],y)
    Q_plot = Q_plot.block_until_ready()
    end = time.time()
    print("Initial estimation took {} seconds".format(round(end-start,2)))

    ## Predictive resampling
    seed = 1706
    B = 5000
    key = jax.random.key(seed)
    key = jax.random.split(key, B)

    #Exact sampling

    # #GP sampling
    ### Approximate Sampling ###
    # Not using JAX as regular scipy has fast multivariate normal
    def approx_PR_B_func(seed,Q_init,a,c,k,n,B,du = 0.005):
        np.random.seed(seed)

        #Initialize grid
        u_plot = jnp.arange(du, 1, du)
        n_plot = np.shape(u_plot)[0]

        #Compute bandwidth
        rho_end = np.sqrt(1 - c*(n+1)**(-k))
        cov = np.array([[1,rho_end**2],[rho_end**2,1]])

        #Compute covariance matrix
        zu,zv = np.meshgrid(sp.stats.norm.ppf(u_plot),sp.stats.norm.ppf(u_plot))
        z_plot = np.vstack([zu.ravel(), zv.ravel()]).transpose()
        cop = sp.stats.multivariate_normal.cdf(z_plot,cov = cov).reshape(n_plot,n_plot)
        Sigma = (cop - np.outer(u_plot,u_plot) + 5e-7*np.eye(n_plot))
        
        #Cholesky decomp for sampling
        chol = np.linalg.cholesky(Sigma)

        #Sample GP
        gp_samp_smooth = np.transpose(np.dot(chol, np.random.randn(n_plot,B))) 
        
        #Add to Q_init
        Q_gp_smooth = Q_init.reshape(1,-1) + a.reshape(1,-1)*gp_samp_smooth/np.sqrt(n+1)
        return Q_gp_smooth

    start = time.time()
    Q_pr_gp = approx_PR_B_func(seed,Q_plot,a,c_opt,k,n,B)
    Q_pr_gp_rearr = rearrange_Q_B(Q_pr_gp)
    end = time.time()
    print("Approx sampling samples took {} seconds".format(round(end-start,2)))

    # #Save data
    np.save("../plots/plot_data/Q_pr_gp_func_rearr_n{}".format(n),Q_pr_gp_rearr)

