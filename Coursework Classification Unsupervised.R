#Set working directory
setwd("C:/UOL/Year 3/ST3189 Machine Learning/Coursework/Data")

#Load Library
library(tidyverse)
library(dplyr)
library(caret)
library(caTools)
library(corrplot)
library(factoextra)
library(ISLR)
library(data.table)
library(leaps) #Subsetting x var
library(rpart) #Decision Tree
library(rpart.plot)
library(randomForest)


#Read Csv
df <- read.csv('telecom_customer_churn.csv', stringsAsFactors = T)
df_num <- select_if(df, is.numeric)

#View dimension, structure, summary
dim(df)
summary(df)
str(df)

#Rename columns
colnames(df)
colnames(df) <- gsub("\\.", "_", colnames(df))
colnames(df_num) <- gsub("\\.", "_", colnames(df_num))
colnames(df)
colnames(df_num)

#Rename Join into stay
df$Customer_Status[df$Customer_Status == "Joined"] <- "Stayed"
df$Customer_Status <- factor(df$Customer_Status, levels = c("Stayed","Churned"))
unique(df$Customer_Status)


#Check for missing values
sum(is.na(df))
sum(duplicated(df))
sum(is.na(df_num))
sum(duplicated(df_num))


#Check for distribution
par(mfrow=c(3,5))
for (i in 1:ncol(df_num)){
  hist(df_num[,i], main = colnames(df_num)[i])
}
par(mfrow = c(1,1))

#Overview of data
library(psych)
corPlot(df_num)
colnames(df_num)

#Drop CustomerID, Zipcode, latitude, longitude column, churn category, churn reason
df <- df[, -c(1,6,7,8,9,37,38)]
df_num <- select_if(df, is.numeric)
str(df)
dim(df)
#Coerce to integer for clustering purposes
df$Gender<- as.integer(df$Gender)
df$Gender <- df$Gender - 1
#Female 0, Male 1

#Fill missing values with median 
#Check numeric columns
df_num <- select_if(df, is.numeric)
numeric_col <- sapply(df, is.numeric)
# Replace missing values with median for numeric columns
df[, numeric_col] <- lapply(df[, numeric_col], function(x) {
  if (any(is.na(x))) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  }
  return(x)
})

df_num <- df_num %>%
  mutate_all(~ if_else(is.na(.), median(., na.rm = TRUE), .))

#Check missing values after patch
sum(is.na(df))
sum(is.na(df_num))

str(df)
summary(df)
# dev.off()
boxplot(df_num)

#Remove outlier function
outliers <- function(x) {
  
  Q1 <- quantile(x, probs=.25)
  Q3 <- quantile(x, probs=.75)
  iqr = Q3-Q1
  
  upper_limit = Q3 + (iqr*1.5)
  lower_limit = Q1 - (iqr*1.5)
  
  x > upper_limit | x < lower_limit
}

remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]),]
  }
  df
}

df <- remove_outliers(df, c('Total_Revenue'))
df_num <- select_if(df, is.numeric)

#Boxplot after outlier fix
boxplot(df_num)


#PCA
pc<- prcomp(df_num, scale.=T)
summary(pc)
pc$rotation
# First 6 components capture 76% of variance 
#PC1: Tenure, total charges, total long distance charges, total revenue
#PC2: Age
#PC3: Avg_Monthly_Long_Distance_Charges, Total_Long_Distance_Charges
#PC4: Avg_Monthly_GB_Download, Monthly_Charge
#PC5: Number of referrals

std_dev <- pc$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

biplot(pc, scale=0)


scaled_data <- scale(df_num)

# K Means Clustering 
#Silhouette method determining number of clusters
fviz_nbclust(scaled_data, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

set.seed(123)
k2 <- kmeans(scaled_data, centers=2)  
summary(k2)

#Visualize
fviz_cluster(k2, data = scaled_data)

k2results <- data.frame(df_num$Gender,df_num$Age, df_num$Total_Revenue, df_num$Tenure_in_Months, k2$cluster)
clust1 <- subset(k2results, k2$cluster==1)
clust2 <- subset(k2results, k2$cluster==2)

#Significant P-value, reject H0 hence all clusters differ in revenue and tenure
anova1 <- aov(df_num.Tenure_in_Months ~ k2.cluster, data = k2results)
summary(anova1)

anova2 <- aov(df_num.Total_Revenue ~ k2.cluster, data = k2results)
summary(anova2)

summary(clust1$df_num.Total_Revenue)
summary(clust2$df_num.Total_Revenue)

#Cluster 1 generates more revenue than cluster 2

table_km <- data.table(Clusters <- c(1, 2), 
                       Observations <- c(n_distinct(clust1), n_distinct(clust2)),
                       Average_Revenue <- c(round(mean(clust1$df_num.Total_Revenue),2), round(mean(clust2$df_num.Total_Revenue),2)))

setnames(table_km, names(table_km), c("Clusters", "Observations", "Average_Revenue"))
table_km


# Hierarchical Clustering 
#Complete and average cannot due to insufficient sample size
#Ward
hc.Ward =hclust(dist(scaled_data), method ="ward.D2")
plot(hc.Ward , main ="Ward Linkage", xlab="", sub ="", cex =.9)
sum(cutree(hc.Ward, 2)==2)  ##2839

hc.cluster1 <- subset(k2results, cutree(hc.Ward, 2)==1)
hc.cluster2 <- subset(k2results, cutree(hc.Ward, 2)==2)

hc.cluster1$df_num.Gender <- factor(hc.cluster1$df_num.Gender)
hc.cluster2$df_num.Gender <- factor(hc.cluster2$df_num.Gender)

summary(hc.cluster1$df_num.Total_Revenue)
summary(hc.cluster2$df_num.Total_Revenue)
## Cluster 2 generates more revenue than cluster 1.


#-------------------------------------------------------------------------------
#Revert back to original datatype
df$Gender <- as.factor(df$Gender)
str(df)

#Train test split
set.seed(2023)
split = sample.split(df$Customer_Status, SplitRatio = 0.8)
train_setcl1 = subset(df, split == TRUE)
test_setcl1 = subset(df, split == FALSE)
summary(train_setcl1)


#Classification ft.scaling
train_setcl1[, c(2,4:6,9,13,25:30)] <- scale(train_setcl1[, c(2,4:6,9,13,25:30)])
test_setcl1[, c(2,4:6,9,13,25:30)] <- scale(test_setcl1[, c(2,4:6,9,13,25:30)])

#Classifier
classifier <- glm(formula = Customer_Status ~.,
                  family = binomial,
                  data = train_setcl1)
#Predict Test set result for clf, baseline is Churn probability
prob_pred <- predict(classifier, type = 'response', newdata = test_setcl1[-31])
y_pred_cl1 <- ifelse(prob_pred > 0.5, 1, 0)
y_pred_cl1
#Confusion Matrix
cm1 <- table(test_setcl1[, 31], y_pred_cl1)
cm1
n_log <- sum(cm1)
diag_cm1 <- diag(cm1)
rowsums_log= apply(cm1, 1, sum) 
colsums_log = apply(cm1, 2, sum) 
accuracy_log <- sum(diag_cm1) / n_log
precision_log = diag_cm1 / colsums_log 
recall_log = diag_cm1 / rowsums_log
f1_log = 2 * precision_log * recall_log / (precision_log + recall_log) 
accuracy_log #85% Accuracy
precision_log

#Evaluate
macroPrecision_log = mean(precision_log)
macroRecall_log = mean(recall_log)
macroF1_log = mean(f1_log)
macro_metrics_log <-  data.frame(accuracy_log,macroPrecision_log, 
                                 macroRecall_log, macroF1_log)
macro_metrics_log
#-------------------------------------------------------------------------------
#Decision Tree
dtree <- rpart(Customer_Status~., 
             data = train_setcl1, 
             method = 'class')

rpart.plot(dtree)


#Predict Dtree
y_pred_dtree = predict(dtree, test_setcl1, type = 'class')
cm2 <- confusionMatrix(factor(test_setcl1$Customer_Status), factor(y_pred_dtree), dnn = c("Prediction", "Actual"))
plt <- as.data.frame(cm2$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Actual,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Prediction",y = "Actual") +
  scale_x_discrete(labels=c("Stayed","Churned")) +
  scale_y_discrete(labels=c("Churned","Stayed"))


cm2

#Evaluate
cm_tree <- table(test_setcl1$Customer_Status, y_pred_dtree)
n_dtree <- sum(cm_tree)
diagonals <- diag(cm_tree)
rowsums_tree= apply(cm_tree, 1, sum) 
colsums_tree = apply(cm_tree, 2, sum) 
accuracy_dtree <- sum(diagonals) / n_dtree 

precision_tree = diagonals / colsums_tree 
recall_tree = diagonals / rowsums_tree
f1_tree = 2 * precision_tree * recall_tree / (precision_tree + recall_tree) 
accuracy_dtree #85% Accuracy
precision_tree

macroPrecision_tree = mean(precision_tree)
macroRecall_tree = mean(recall_tree)
macroF1_tree = mean(f1_tree)
macro_metrics_tree <-  data.frame(accuracy_dtree,macroPrecision_tree, 
                                 macroRecall_tree, macroF1_tree)
macro_metrics_tree


#Random Forest classifier
classifier2 <- randomForest(x = train_setcl1[-31],
                           y = train_setcl1$Customer_Status,
                           ntree = 100)
importance(classifier2)
varImpPlot(classifier2) #Top 3: Contract, Monthly_Charge, Tenure

y_predcl2 <- predict(classifier2, newdata = test_setcl1[-31])
cm3 <- table(test_setcl1[, 31], y_predcl2)
cm3
#Evaluate
cm_rf<- table(test_setcl1$Customer_Status, y_predcl2)
n_rf <- sum(cm_rf)
diag_rf <- diag(cm_rf)
rowsums_rf= apply(cm_rf, 1, sum) 
colsums_rf = apply(cm_rf, 2, sum) 
accuracy_rf <- sum(diag_rf) / n_rf 
cm_rf
precision_rf = diag_rf / colsums_rf
recall_rf = diag_rf / rowsums_rf
f1_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf) 
accuracy_rf #85% Accuracy
precision_rf

macroPrecision_rf = mean(precision_rf)
macroRecall_rf = mean(recall_rf)
macroF1_rf = mean(f1_rf)
macro_metrics_rf <-  data.frame(accuracy_rf,macroPrecision_rf, 
                                  macroRecall_rf, macroF1_rf)
macro_metrics_rf



#Table summary classification
table_class <- data.table(Techniques <- c("Logistic Regression", "Decision Tree", "Random Forest"),
                          Accuracy <- c(macro_metrics_log$accuracy_log, macro_metrics_tree$accuracy_dtree, macro_metrics_rf$accuracy_rf),
                          F1_Score <- c(macro_metrics_log$macroF1_log, macro_metrics_tree$macroF1_tree, macro_metrics_rf$macroF1_rf)) 

setnames(table_class, names(table_class), c("Techniques", "Accuracy", "F1 Scores"))
setorder(table_class, cols = - "Accuracy")
table_class

