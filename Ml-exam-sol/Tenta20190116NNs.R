# JMP

library(neuralnet)
set.seed(1234567890)

Var <- runif(50, 0, 3)
tr <- data.frame(Var, Sin=sin(Var))
Var <- runif(50, 3, 9)
te <- data.frame(Var, Sin=sin(Var))
winit <- runif(10, -1, 1)
nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 3, startweights = winit, lifesign = "full")
plot(tr,xlim=c(0,9),ylim=c(-2,2))
points(te,col="blue")
points(te$Var,compute(nn,te$Var)$net.result,col="red")
plot(nn)

# The predictions converge to -2 because the sigmoid functions saturate for large Var values.