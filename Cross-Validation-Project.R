########################
# Eduardo Guzman  
# Predictive Modeling 
# Spring 2022 
########################


########################
###      Part 1      ### 
########################

#import the dataset 
mydata <- read.table("project2.data", sep = ';', header = T)

#Observe that we have some missing values only in the X3 and Y columns
which(is.na(mydata$x1))
which(is.na(mydata$x2))
which(is.na(mydata$x3))
which(is.na(mydata$x4))
which(is.na(mydata$x5))
which(is.na(mydata$y))

#Collct the indices of the observations with NA values. 
removeindex <- NULL
removeindex <- append(removeindex, which(is.na(mydata$x3)))
removeindex <- append(removeindex, which(is.na(mydata$y)))

#Observe that there are 20 rows of data that we will need to remove
removeindex <- unique(removeindex)
length(removeindex)

#Update the dataset by remove those indices 
mydata <- mydata[-removeindex,]


########################
###      Part 2      ### 
########################
set.seed(2022)
n <- dim(mydata)[1]

#Create a 70-30 split between train and test data 
id <- sample(rep(1:10, length=n))
train <- mydata[which(id<=7), ]
test <- mydata[which(id>=8), ]


########################
###      Part 3      ### 
########################
pairs(train, upper.panel = NULL, pch=16, col=train$y+1)

#Calculate the mean of each varaible for each subgroup 
aggregate(train$x1, list(train$y), FUN=mean) # 0: -0.1217   1: -1.0299
aggregate(train$x2, list(train$y), FUN=mean) # 0:  0.0030   1: -0.0240
aggregate(train$x3, list(train$y), FUN=mean) # 0:  0.1182   1:  1.0631
aggregate(train$x4, list(train$y), FUN=mean) # 0: -0.0920   1: -0.0920
aggregate(train$x5, list(train$y), FUN=mean) # 0: -0.0246   1: -0.0220


#Standard deviations for each variable where y=1 (account defaults)
sd(train$x1[train$y==1]) #1.0013
sd(train$x2[train$y==1]) #1.2103
sd(train$x3[train$y==1]) #1.2234
sd(train$x4[train$y==1]) #0.9198
sd(train$x5[train$y==1]) #0.9106
sd(train$y[train$y==1])  #0.0000

#Standard deviations for each variable where y=0 (non-default)
sd(train$x1[train$y==0]) #0.9706
sd(train$x2[train$y==0]) #0.9327
sd(train$x3[train$y==0]) #0.9442
sd(train$x4[train$y==0]) #1.0249
sd(train$x5[train$y==0]) #1.0572
sd(train$y[train$y==0])  #0.0000




########################
###     Part A1      ### 
########################

#Split the dataset into 8 folds
n2 <- dim(train)[1]
folds <- sample(rep(1:8, length=n2))
folds[1:10]

#Run 8-Fold Cross validation on the models
Err.mod1 <- NULL
Err.mod2 <- NULL
for (k in 1:8){
  #Define the test/train sets for this iteration 
  cvtest <- train[folds==k,]
  cvtrain <- train[folds!=k,]
  
  #Train a Logistic Regression Model with linear terms only (drop x4 and x5)
  mod1 <- glm(y~x1+x2+x3, data=cvtrain, family="binomial")
  mod2 <- glm(formula = y ~ poly(x1, degree = 2)+x2+x3, data = cvtrain, family = "binomial") 
  
  #Get the predicted probabilities and classification for the test set
  mod1.probs <- predict.glm(mod1, newdata = cvtest, type = "response")
  mod1.preds <- as.numeric(mod1.probs >= 0.5)
  
  mod2_probs <- predict.glm(mod2, newdata = cvtest, type = "response")
  mod2_predict <- as.numeric(mod2_probs >= 0.5)
  
  
  #Count the number of occurrences where the predicted class does not match the observed class
  len <- length(mod1.probs)
  mod1.errors = 0 
  mod2.errors = 0
  for(i in 1:len){
    if(mod1.preds[i] != cvtest$y[i]){
      mod1.errors = mod1.errors + 1 
    }
    if(mod2_predict[i] != cvtest$y[i]){
      mod2.errors = mod2.errors + 1 
    }
  }
  
  #Record the misclassification rates of each model for this iteration
  Err.mod1[k] <- (mod1.errors / len)
  Err.mod2[k] <- (mod2.errors / len)
}


#Get the CV for the Linear Regression Model with linear terms 
nk <- 42 
n <- 336

CV.mod1 <-  sum((nk/n) * Err.mod1)
CV.mod1 #0.25

CV.mod2 <- sum((nk/n) * Err.mod2)
CV.mod2 #0.217


#Observe that the model with the quadratic term yielded the lower CV score. 
#This suggests that this model is better at making predictions 





########################
###     Part 5      ### 
########################

#Fit the selected model on the full training data 
mod <- glm(y ~ poly(x1, degree = 2)+x2+x3, data = train, family = "binomial") 

#Use this model to evaluate the predicted values in the validation set
probs <- predict.glm(mod, newdata = test, type = "response")
preds <- as.numeric(probs >= 0.5)

table(preds, test$y)

#Overall Misclassification Rate
(8 + 21) / 144  #0.2013889

#False Positive Rate
21 / (21 + 55) #0.2763

#False Negative Rate
8 / (8 + 60) #0.1176

