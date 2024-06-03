import numpy as np
import scipy as sp
from functools import partial

#import jax functions
import jax.numpy as jnp
from jax import grad,value_and_grad, jit, vmap,jacfwd,jacrev,random
from jax.scipy.stats import norm
from jax.lax import fori_loop,scan

#import package functions
from . import qmp_functions as qmp
from .utils.bivariate_copula import ndtri_

from .utils.bivariate_copula import log_Huv,ndtri_

### Predictive resampling functions ###

#### Main function ####
#updating without rearrangement nor prequential score
@jit
def update_Q_norearr(carry,i):
    Q_plot,u_plot,vT,a,c,k = carry

    #Update quantile 
    alpha = a/((i+2))
    rho = jnp.sqrt(1-c/((i+1)**(k)))
    #rho = k
    Q_plot = Q_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,vT[i],rho)))

    carry = Q_plot,u_plot,vT,a,c,k
    return carry,i

#Scan over y_{1:n}
@jit
def update_Q_norearr_scan(carry,rng):
    return scan(update_Q_norearr,carry,rng)


# Loop through forward sampling; generate uniform random variables, then use p(y) update from mvcd
@partial(jit,static_argnums = (5,6))
def PR_loop(key,Q_init,a,c,k,n,T,du = 0.005):

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,))

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.append(jnp.zeros((n)),a_rand)

    #Initialize grid
    u_plot = jnp.arange(du, 1, du)

    #run forward loop
    inputs = Q_init,u_plot,vT,a,c,k
    rng = jnp.arange(n,n+T)
    outputs,rng = update_Q_norearr_scan(inputs,rng)
    Q_plot,*_ = outputs

    return Q_plot

## Vmap over multiple test points, then over multiple seeds
PR_loop_B =jit(vmap(PR_loop,(0,None,None,None,None,None,None)),static_argnums = (2,3,4,5,6)) #vmap across B posterior samples
#### ####

#### Convergence checks ####

# Update p(y) in forward sampling, while keeping a track of change in p(y) for convergence check
def pr_1step_conv(i,inputs):  #t = n+i
    Q_plot,u_plot,vT,a,k, Q_init,Qdiff = carry

    #Update quantile 
    alpha = a/((i+2))
    rho = jnp.sqrt(1-1/((i+2)**(k)))
    Q_plot = Q_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,vT[i],rho)))

    #Compute L2 difference
    Qdiff = Qdiff.at[i].set(jnp.mean(jnp.abs(Q_plot- Q_init)**2)) #mean L2 difference

    carry = Q_plot,u_plot,vT,a,k, Q_init,Qdiff    
    return carry


### Approximate Sampling ###
# Not using JAX as regular scipy has fast multivariate normal
def approx_PR_B(seed,Q_init,a,c,k,n,B,du = 0.005):
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
    Q_gp_smooth = Q_init.reshape(1,-1) + a*gp_samp_smooth/np.sqrt(n+1)
    return Q_gp_smooth
