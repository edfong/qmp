library(quantreg)
library(tidyr)
library(dplyr)
library(fastDummies)
df <- read.csv('../data/globalTCmax4.txt', sep = " ") %>% 
  mutate(Basin = replace_na(Basin,"NoA"))  %>%
  filter(Basin != 'SA') 

df_ = df %>% dplyr::select(WmaxST,Year, Age,Lat) %>% scale %>% data.frame 

tau_plot = seq(0.005,0.995,0.005)
n_plot = length(tau_plot)
fit = list()

beta = matrix(nrow = n_plot,ncol = dim(df_)[2])

y = df_ %>% dplyr::select(WmaxST) %>% pull
x = data.matrix(df_ %>% dplyr::select(-WmaxST))

for (i in 1:n_plot){
  fit[[i]] <- rq(WmaxST ~ .,data = df_,tau = tau_plot[i])
  beta[i,] <- coef(fit[[i]])
}

write.csv(beta,'../plots/plot_data/beta_big.csv')
write.csv(tau_plot,'../plots/plot_data/tau_plot.csv')
