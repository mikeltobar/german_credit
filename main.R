## ----setup, include=FALSE------------------------
knitr::opts_chunk$set(echo = TRUE)


## ---- warning=F, message=F-----------------------
library("ggplot2")
library("miscset")
library("Boruta")
library("dplyr")
library("e1071")
library("caret")
library("randomForest")
library("rpart")


## ------------------------------------------------
credit <-read.csv("credit_kaggle.csv", TRUE, ",", fileEncoding = "UTF-8")
head(credit,10)


## ------------------------------------------------
summary(credit)
str(credit)


## ------------------------------------------------
credit$default<-gsub(1,0,credit$default)
credit$default<-gsub(2,1,credit$default)
credit$dependents<-gsub(1,0,credit$dependents)
credit$dependents<-gsub(2,1,credit$dependents)


## ----fig.height=10, fig.width=12-----------------
ggplotGrid(ncol = 4,
  lapply(c("checking_balance", "credit_history", "purpose", "savings_balance", "employment_length", "personal_status", "other_debtors", "property", "installment_plan", "housing", "telephone", "foreign_worker", "job", "installment_rate","existing_credits", "residence_history"),
    function(col) {
        ggplot(credit, aes_string(col)) + geom_bar() + coord_flip()
    }))


## ------------------------------------------------
hist(credit$months_loan_duration,xlab="Credit length (months)", ylab="Clients", main="Clients according to credit duration (in months)")
hist(credit$amount,xlab="Importe", ylab="Clients", main="Clients according to credit amount")
hist(credit$age,xlab="Age", ylab="Clients", main="Clients by age")


## ------------------------------------------------
boxplot(credit$months_loan_duration, main="Credit duration (months)")
boxplot(credit$amount, main="Amount of the credit given")
boxplot(credit$age, main="Clients' age")


## ------------------------------------------------
cols <- c("checking_balance", "credit_history", "purpose", "savings_balance", "employment_length","personal_status","other_debtors","property","installment_plan","housing","default","telephone","foreign_worker","job")

credit[,cols] <- lapply(credit[,cols],as.factor)


credit$checking_balance <- factor(credit$checking_balance)
glm.credit<- glm(default~., family=binomial, data=credit)
summary(glm.credit)

exp(coefficients(glm.credit))



## ----message=F, warning=F------------------------
fit <- Boruta(Species ~ ., data = iris, doTrace = 2);
boruta.credit_train <- Boruta(default~., data = credit, doTrace = 2)


## ------------------------------------------------
print(boruta.credit_train)

par(mar=c(10,5,5,5)+.1)
plot(boruta.credit_train, xlab= "", las=3)


## ------------------------------------------------
credit<-subset(credit, select=(c(1:6,8,10,12:14,16,17)))  


## ------------------------------------------------
set.seed(1432)
y <- credit[,13] 
x <- credit[,1:12] 


## ----fig.height=15, fig.width=18-----------------
split_prop <- 3
max_split<-floor(nrow(x)/split_prop)
tr_limit <- nrow(x)-max_split
ts_limit <- nrow(x)-max_split+1

trainx <- x[1:tr_limit,]
trainy <- y[1:tr_limit]
testx <- x[ts_limit+1:nrow(x),]
testy <- y[ts_limit+1:nrow(x)]

split_prop <- 3
indexes = sample(1:nrow(credit), size=floor(((split_prop-1)/split_prop)*nrow(credit)))
trainx<-x[indexes,]
trainy<-y[indexes]
testx<-x[-indexes,]
testy<-y[-indexes]

summary(testx)
summary(testy)
summary(trainx)
summary(trainy)

trainy = as.factor(trainy)
model <- C50::C5.0(trainx, trainy,rules=TRUE )
summary(model)

model <- C50::C5.0(trainx, trainy)
plot(model)


## ------------------------------------------------
predicted_model <- predict( model, testx, type="class" )
print(sprintf("The tree's precission is: %.4f %%",100*sum(predicted_model == testy) / length(predicted_model)))


## ------------------------------------------------
mat_conf<-table(testy,Predicted=predicted_model)
mat_conf


## ------------------------------------------------
porcentaje_correct<-100 * sum(diag(mat_conf)) / sum(mat_conf)
print(sprintf("The percentage of results correctly classified is: %.4f %%",porcentaje_correct))


## ------------------------------------------------
if(!require(gmodels)){
    install.packages('gmodels', repos='http://cran.us.r-project.org')
    library(gmodels)
}

CrossTable(testy, predicted_model,prop.chisq  = FALSE, prop.c = FALSE, prop.r =FALSE,dnn = c('Reality', 'Prediction'))


## ------------------------------------------------
modelo<-randomForest(default~., data=credit, proximity=T)
library("randomForest")

modelo<-randomForest(trainx, trainy, proximity=T)
modelo

predicted_model1 <- predict( modelo, testx, type="class" )
print(sprintf("The Random Forest's accuracy is: %.4f %%",100*sum(predicted_model1 == testy) / length(predicted_model1)))

mat_conf1<-table(testy,Predicted=predicted_model1)
mat_conf1


## ------------------------------------------------
modelo2<-naiveBayes(trainx, trainy, proximity=T)
modelo2

predicted_model2 <- predict( modelo2, testx, type="class" )
print(sprintf("The Bayesian Model's accuracy is: %.4f %%",100*sum(predicted_model2 == testy) / length(predicted_model2)))

mat_conf2<-table(testy,Predicted=predicted_model2)
mat_conf2

