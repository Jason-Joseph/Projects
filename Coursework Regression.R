#Set wd()
setwd("C:/UOL/Year 3/ST3189 Machine Learning/Coursework/Data")

#Load Library
library(tidyverse)
library(dplyr)
library(caret)
library(caTools)
library(data.table)
library(corrplot)
library(rpart) #Decision Tree
library(rpart.plot)
library(randomForest)
library(leaps)
library(psych)


#Read CSV
df <- read.csv("CarPrice_Assignment.csv")
head(df)
#Drop column car_ID and CarName
df <- df[,-c(1, 3)]
#Check for duplicates and remove
sum(duplicated(df))
df <- df[!duplicated(df), ]
sum(duplicated(df))
#Overview
dim(df)
str(df)
summary(df)
sum(is.na(df))

#Select Numerical
df_num <- select_if(df, is.numeric)
#Corrplot
corPlot(df_num)
#Check for outlier
boxplot(df_num)


#Remove outlier function
outliers <- function(x) {
  Q1 <- quantile(x, probs = .25)
  Q3 <- quantile(x, probs = .75)
  iqr = Q3 - Q1
  
  upper_limit = Q3 + (iqr * 1.5)
  lower_limit = Q1 - (iqr * 1.5)
  
  x > upper_limit | x < lower_limit
}

remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]), ]
  }
  df
}


df2 <- remove_outliers(df, c('price'))
df_num2 <- select_if(df2, is.numeric)

#Check outlier after fix
boxplot(df_num2)

#Distribution
par(mfrow = c(4, 4))
for (i in 1:ncol(df_num2)) {
  hist(df_num2[, i], main = colnames(df_num2)[i])
}
par(mfrow = c(1, 1))


one_distinct <- sapply(df2, function(x)
  n_distinct(x) == 1)
#Print column names with only one distinct value
names(df2)[one_distinct]

# #Drop engine location as it only has 1 factor value
df2 <- subset(df2, select = -c(enginelocation))
#Drop cylindernumber three as only 1 element to avoid issues of traintest split
df2 <- df[-19, ]
#Train test split
set.seed(123)
split = sample.split(df2$price, SplitRatio = 0.8)
train_set = subset(df2, split == TRUE)
test_set = subset(df2, split == FALSE)

#MODEL 1:  Multiple Linear Regression
m1_train <- lm(price ~ ., data = train_set)
summary(m1_train)

m1_test <- lm(price ~ ., data = test_set)
summary(m1_test)
#Diagnostic Plot
par(mfrow = (c(2, 2)))
plot(m1_train)
par(mfrow = c(1, 1))

#Stepwise backward elimination
stepw1_train <- step(m1_train, direction = "backward")
summary(stepw1_train)

stepw1_test <- step(m1_test, direction = "backward")
summary(stepw1_test)
#RMSE
rmse_m1_train <- round(sqrt(mean(residuals(stepw1_train) ^ 2)), 2)
rmse_m1_test <- round(sqrt(mean(residuals(stepw1_test) ^ 2)))
#Rsquared
mlr_rsq <- summary(stepw1_train)$adj.r.squared


#MODEL 2: CART
#CART
cart <-
  rpart(
    price ~ .,
    data = train_set,
    method = 'anova',
    control = rpart.control(minsplit = 2, cp = 0)
  )
rpart.plot(cart, nn = T, main = "Maxmimal Tree in CART")
printcp(cart, digits = 3)
plotcp(cart)

#Extract optimal tree
CVerror.cap <-cart$cptable[which.min(cart$cptable[, "xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[, "xerror"]), "xstd"]
# CP Region below horizontal line
i <- 1
j <- 4
while (cart$cptable[i, j] > CVerror.cap) {
  i <- i + 1
}
#9th tree is the optimal tree based on 1SE rule

#Geometric mean
cp.opt = ifelse(i > 1, sqrt(cart$cptable[i, 1] * cart$cptable[i - 1, 1]), 1)

prune_cart <- prune(cart, cp = cp.opt)

rmse_cart.train <-
  round(sqrt(mean((
    train_set$price - predict(prune_cart)
  ) ^ 2)))
rmse_cart.test <-
  round(sqrt(mean((
    test_set$price - predict(prune_cart, newdata = test_set)
  ) ^ 2)))

cart$variable.importance
#Enginesize, curbweight, horsepower are important in explaining car price
cart_pred = predict(cart, test_set)
sst <- sum((test_set$price - mean(test_set$price)) ^ 2)
sse <- sum((cart_pred - test_set$price) ^ 2)
#Find R-Squared
cart_rsq <- 1 - sse / sst



#Random Forest
set.seed(345)
rf_reg <- randomForest(
  price ~ .,
  data = train_set,
  ntree = 500,
  keep.forest = FALSE,
  importance = TRUE
)
rf_reg

# Calculate RMSE
rmse_rf <- round(sqrt(rf_reg$mse[length(rf_reg$mse)]), 2)

# Calculate R-squared
rf_rsq <- rf_reg$rsq[length(rf_reg$rsq)]

# Get variable importance from the model fit
mfit <- as.data.frame(importance(rf_reg))
mfit$Var.Names <- row.names(mfit)

ggplot(mfit, aes(x = Var.Names, y = `%IncMSE`)) +
  geom_segment(aes(
    x = Var.Names,
    xend = Var.Names,
    y = 0,
    yend = `%IncMSE`
  ),
  color = "skyblue") +
  geom_point(aes(size = IncNodePurity),
             color = "blue",
             alpha = 0.6) +
  theme_light() +
  coord_flip() +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )
#Engine size, curbweight, horsepower




#Table summary Regression
table_class <-data.table(Techniques <-c("Multiple Linear Regression","CART Model","Random Forest Regression"),
                         RSquared <- c(mlr_rsq, cart_rsq, rf_rsq),
                         RMSE <-c(rmse_m1_test, rmse_cart.test, rmse_rf))

setnames(table_class,names(table_class),c("Techniques", "R-Squared", "RMSE"))
setorder(table_class, cols = "RMSE")
table_class
