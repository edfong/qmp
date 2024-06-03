library(quantreg)
df <- read.csv('../data/globalTCmax4.txt', sep = " ")
df = df[is.na(df$Basin),]
y = df$WmaxST
x = df$Year
y = (y - mean(y))/sd(y)
x = (x - mean(x))/sd(x)

tau_plot = seq(0.005,0.995,0.005)
n_plot = length(tau_plot)
intercept = rep(0,n_plot)
beta = rep(0,n_plot)
low = rep(0,n_plot)
high = rep(0,n_plot)

for (i in 1:n_plot){
  fit <- rq(y~x,tau = tau_plot[i])
  intercept[i] <- coef(fit)[1]
  beta[i] <- coef(fit)[2]
  boot_samp <- boot.rq(cbind(1,x),y,tau = tau_plot[i])$B
  low[i] <- quantile(boot_samp[,2],0.025)
  high[i] <- quantile(boot_samp[,2],0.975)
}

write.csv(intercept,'../plots/plot_data/intercept_small.csv')
write.csv(beta,'../plots/plot_data/beta_small.csv')
write.csv(tau_plot,'../plots/plot_data/tau_plot.csv')
