import jax
import jax.numpy as jnp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from tqdm import tqdm_notebook
import pandas as pd
from qmp.qmp_reg_functions import fit_beta, fit_beta_perm
from qmp.sample_qmp_reg_functions import approx_PR_reg_B, PR_reg_loop_B
from qmp.qmp_functions import rearrange_Q
from qmp.qmp_functions import rearrange_Q_B

# Load data ####
df = pd.read_csv("../data/globalTCmax4.txt", sep = " ",keep_default_na = False)
df = df[df['Basin']!= 'SA']
n = np.shape(df)[0]
x = df[['Year','Age','Lat']]
y = np.array(df['WmaxST'])

# Initialize plots grids ####
du = 0.005
u_plot = jnp.arange(du, 1, du)
dx = 0.01
x_plot = jnp.arange(dx,1,dx)


# Standardize ####
mean_y = np.mean(y)
sd_y = np.std(y)
y = (y-mean_y)/sd_y

mean_x = np.mean(x,axis = 0) 
sd_x = np.std(x,axis = 0)
x = (x-mean_x)/sd_x


# Add intercept ####
x = np.hstack((np.ones(n).reshape(-1,1),x)) 
d = np.shape(x)[1]

# Estimate hyperparameters ####
start = time.time()

# Estimate a ####
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
sigma = np.sqrt(np.mean((y - model.predict(x))**2)) #comptue variance of residuals

a = np.sqrt(12)*sigma/(np.linalg.det(np.dot(np.transpose(x),x)/n)) 
print('a = {}'.format(round(a,2)))
k = 0.5

# Estimate c ####
#Fit
c_vals = np.arange(0.05,1.0,0.05)
n_c = len(c_vals)
preq_score = np.zeros(n_c)

for i in range(n_c):
    #Fit
    c = c_vals[i]
    Q_plot,preq_score_vec = fit_beta_perm(np.array([a,c,k]),y,x,n_perm = 10)
    preq_score[i] = np.mean(preq_score_vec)

#plt.figure()
c_preq = c_vals[np.argmax(preq_score)]
c_opt = c_preq
print('c = {}'.format(round(c_opt,2)))
end = time.time()
print('Hyperparameter estimation took {} seconds'.format(round(end - start),2))

# Estimation
start = time.time()
beta_plot,preq_score_vec = fit_beta_perm(np.array([a,c_opt,k]),y,x,n_perm = 10)
beta_plot = beta_plot.block_until_ready()
end = time.time()
print('Initial estimation took {} seconds'.format(round(end - start,2)))

# Exact sampling
seed = 1706
B = 10000
key = jax.random.key(seed)
key = jax.random.split(key, B)
T = 5000
start = time.time()
beta_pr_exact = PR_reg_loop_B(key,jnp.array(beta_plot),jnp.array(x),a,c_opt,k,n,T)
end = time.time()
print('Exact sampling took {} seconds'.format(round(end - start,2)))

# Save
np.save("../plots/plot_data/beta_pr_exact_n{}".format(n),beta_pr_exact)

# GP approximate sampling
seed = 5124
B = 10000
start = time.time()
beta_pr = approx_PR_reg_B(seed,beta_plot,x,a,c_opt,k,n,B)
end = time.time()
print('GP sampling took {} seconds'.format(round(end - start,2)))

# Save
np.save("../plots/plot_data/beta_pr_n{}".format(n),beta_pr)

# Save hyperparameters
hyperparam_opt = {'a' : a, 'c': c_opt, 'k': k,  'd':d, 'B': B}
np.save("../plots/plot_data/hyperparam_reg_opt_n{}".format(n),hyperparam_opt)

# Save data and normalizing constants
np.save("../plots/plot_data/data_reg_n{}".format(n),{'x' : x, 'y': y})
np.save("../plots/plot_data/normalize_reg_n{}".format(n),{'mean_x' : mean_x, 'sd_x': sd_x,'mean_y' : mean_y, 'sd_y': sd_y})