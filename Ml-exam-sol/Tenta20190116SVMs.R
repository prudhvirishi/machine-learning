# JMP

library(kernlab)
set.seed(1234567890)

data(spam)

index <- sample(1:4601)
tr <- spam[index[1:2500], ]
va <- spam[index[2501:3500], ]
te <- spam[index[3501:4601], ]                      
                         
# My first guess was that you may get an error ("No svms found") when C=0 because your data may not be linearly
# separable. However, one gets the error even when the data is linearly separable, e.g.

ksvm(type~.,data=tr[1:2,57:58],kernel="rbfdot",kpar=list(sigma=0.05),C=0)

# It is true that theory requires C>0 but, then, I would have expected an information message telling me to enter
# a positive C, not an error. Bottom line: I am not sure why you get an error but I do not think it is because
# C must be positive. Issuing an error would be a weird way to instruct the user to raise the value of C. I had
# expected that C=0 instruct the system to try to find a linearly separable SVM.

# Model selection.

er<-NULL
myStep<-0.1
for(myC in seq(0.1,10,myStep)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=myC)
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  er<-c(er,(t[1,2]+t[2,1])/sum(t))
}
plot(er)

# Final model.

min(er)
which.min(er)
filter<-ksvm(type~.,data=spam[index[1:3500],],kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(er)*myStep)
mailtype <- predict(filter,te[,-58])
t <- table(mailtype,te[,58])
(t[1,2]+t[2,1])/sum(t)
filter<-ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(er)*myStep)

# That the validation and test error are similar suggests no overfitting.

# Pseudocode.

rbfkernel <- rbfdot(sigma = 0.05)
sv<-alphaindex(filter)[[1]]
co<-coef(filter)[[1]]
k<-NULL
for(i in 1:1000){
  k2<-NULL
  for(j in 1:length(sv)){
    k2<-c(k2,co[j]*rbfkernel(unlist(tr[sv[j],-58]),unlist(va[i,-58])))
    }
  k<-c(k,sum(k2)-b(filter))
}
k
