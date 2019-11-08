# Manipulate for mixture models
# Author: Jose M. Peña, STIMA, Linkoping University, Sweden. e-mail: jose.m.pena@liu.se

install.packages("manipulate")
library(manipulate)

data(faithful)
hist(faithful$waiting,freq = FALSE)

NormalPlot <- function(mu1,sigma1,mu2,sigma2,pi1){
  xGrid <- seq(40, 100, by=0.001)
  pdf = dnorm(xGrid, mean=mu1, sd=sigma1) * pi1 + (1-pi1) * dnorm(xGrid, mean=mu2, sd=sigma2)
  hist(faithful$waiting,freq = FALSE)
  lines(xGrid, pdf, type = 'l', lwd = 3, col = "blue")
}

manipulate(
  NormalPlot(mu1,sigma1,mu2,sigma2,pi1),
  mu1 = slider(40, 100, step=.1, initial = 70, label = "mu1"),
  sigma1 = slider(0, 10, step=.1, initial = 1, label = "sigma1"),
  mu2 = slider(40, 100, step=.1, initial = 70, label = "mu2"),
  sigma2 = slider(0, 10, step=.1, initial = 1, label = "sigma2"),
  pi1 = slider(0, 1, step=.01, initial = .5, label = "pi1")
)

