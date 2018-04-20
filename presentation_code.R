# k-Nearest Neighbors analysis on biometric walking data
# Jake Catlett
# June 15, 2016
# jfcatlett74@gmail.com

# The data for this analysis was downloaded at the following site, and is
# available for public use:
# https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity

# first, set the directory in quotes so that it points to the directory
# where the data is stored

setwd("~/Documents/KNN_Presentation")

#import the first of 22 datasets containing biometric walking data

walkData <- read.csv("1.csv", header = FALSE)

# View the first 6 observations of one of our .csv files

head(walkData)

# Note that more or less than 6 observations can be viewed by setting the
# n parameter

head(walkData, n=10)

# add a column to the dataset which indicates which class the records
# belong to - in our case the number of the dataset pulled from 
# will suffice, then view the head of the data again to make sure the
# operation executed as expected

walkData$class <- as.factor(1)
head(walkData)

# the following loop opens each of the remaining 21 datasets, stores
# them in a temporary data frame, adds a row to contain the
# classifier for each set, then binds them to 
# the data frame being used to store the data

for (i in 2:22) {
  fileName <- paste(toString(i), ".csv", sep = "")
  tempFrame <- read.csv(fileName, header = FALSE)
  tempFrame$class <- as.factor(i)
  walkData <- rbind(walkData, tempFrame)
}

# The raw dataset is then saved as a .csv file
# in the working directory.

write.csv(walkData, 'walk_data.csv', row.names = FALSE)

# It is encouraged to open the file created in a spreadsheet or in some other format
# in order to verify that it has saved in the format expected.

# View the head, tail, and structure of the new data frame to verify that it has
# the expected columns and to analyze its contents

head(walkData)
tail(walkData)
str(walkData)

# Change column names to reflect their meanings

colnames(walkData) <- c("step_time", "acc_x", "acc_y", "acc_z", "class")

# View a statistical summary of the data

summary(walkData)

# Explore the proportion of records assigned to each class

class_proportions <- round(prop.table(table(walkData$class)) * 100, digits = 1)
class_proportions

# View a bar plot of the distrubution of the classes in the dataset.  Note that
# the object class_proportions can simply be fed to the barplot() function
# as a parameter.  Oh, R, you're so nifty.

barplot(class_proportions,
        col = "blue",
        border = "green",
        main = "Distribution by Classes",
        ylab = "Proportion",
        xlab = "Class")

# Create a matrix of 4 boxplots which show the distribution of data in each of
# the four variables in the set.  The settings in the par() function create a 
# matrix with 1 row and 4 columns, and the 4 boxplots created after that then 
# get placed into the matrix.  This ensures that the graphic produced is a single
# image with 4 boxplots stacked horizontally.  Without the par() function run first
# the boxplots will be created as four separate images.

par(mfcol = c(1, 4))
boxplot(walkData$step_time, main = "Step Time", 
        col = "lightblue", outcol = "red")
boxplot(walkData$acc_x, main = "x Acceleration", 
        col = "lightblue", outcol = "red")
boxplot(walkData$acc_y, main = "y Acceleration", 
        col = "lightblue", outcol = "red")
boxplot(walkData$acc_z, main = "z Acceleration", 
        col = "lightblue", outcol = "red")

# Create script to normalize data, then create a data frame of normalized dependent
# variables

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

walkData_N <- as.data.frame(lapply(walkData[1:4], normalize))

# Summarize normalized data frame to check results

summary(walkData_N)

# Create data frame of dependent variables scaled to z-scores

walkData_Z <- as.data.frame(scale(walkData[1:4]))

# Summarize scaled data frame to check results

summary(walkData_Z)

# The seed is set here to ensure that the sample will remain the same every time
# the code is run.  If you wish, you can change the number in the set.seed() function
# to an arbitrary value, but be aware that this will result in a different sample
# set for the training and test sets, and therefore slightly different results at
# the end of the analysis.

set.seed(1212)

# Create an index to use to create a random sample for train and test sets
# which preserves the proportions of the class variable found in the dataset using
# the sample.split() function in the caTools package.  Then create train and test 
# sets from the normalized and scaled data frames.  If you do not have the caTools 
# package intalled yet, don't forget to do so by using install.packages("caTools)

library(caTools)

training_split <- sample.split(walkData$class, SplitRatio = 9/10)
normal_train <- walkData_N[training_split, ]
normal_test <- walkData_N[!training_split, ]
Z_train <- walkData_Z[training_split, ]
Z_test <- walkData_Z[!training_split, ]

# Class labels for the train and test sets imported

walkTrain_labels <- walkData$class[training_split]
walkTest_labels <- walkData$class[!training_split]

# Check that the proportions of the class variable are the same for both sets

round(prop.table(table(walkTrain_labels)) * 100, digits = 1)
round(prop.table(table(walkTest_labels)) * 100, digits = 1)

# Create two barplots to confirm that the proportionality of the classes within the
# dataset is preserved by the data split.

par(mfcol = c(1, 1))

barplot(round(prop.table(table(walkTrain_labels)) * 100, digits = 1),
        col = "blue",
        border = "green",
        main = "Class Distributions in Training Set",
        ylab = "Proportion",
        xlab = "Class")

barplot(round(prop.table(table(walkTest_labels)) * 100, digits = 1),
        col = "blue",
        border = "green",
        main = "Class Distributions in Test Set",
        ylab = "Proportion",
        xlab = "Class")

# Find the number of rows in each dataset that contain the most under-represented
# class, 19.  Then determine the nearest integer to the square root of those numbers 
# to test as values for the k parameter

sum(walkTrain_labels == 19)
sum(walkTest_labels == 19)

round(sqrt(sum(walkTrain_labels == 19)))
round(sqrt(sum(walkTest_labels == 19)))

# Run the k-NN algorithm and create a list of predictions for the test set with
# the parameters k = 10 and k = 30 using normalized data.  The function knn() is found 
# in the class package, and so the class must be installed and loaded before 
# it can be used.

library(class)

walkTest_N_pred_k10 <- knn(train = normal_train, 
                           test = normal_test,
                           cl = walkTrain_labels,
                           k = 10)

walkTest_N_pred_k30 <- knn(train = normal_train, 
                           test = normal_test,
                           cl = walkTrain_labels,
                           k = 30)

# Then repeat the process to make two sets of predictions using data
# that has been scaled to z-scores

walkTest_Z_pred_k10 <- knn(train = Z_train, 
                           test = Z_test,
                           cl = walkTrain_labels, 
                           k = 10)

walkTest_Z_pred_k30 <- knn(train = Z_train, 
                           test = Z_test,
                           cl = walkTrain_labels, 
                           k = 30)

# For each set of predictions, create an index of records that match between 
# the predictions and the list of true classes for the test sets.  Divide the length
# of this index (i.e. the number of records observations is contains) by the number 
# of rows in the test label set to determine the accuracy rate.

matches <- which(walkTest_labels == walkTest_N_pred_k10)
round((length(matches) / length(walkTest_labels)) * 100, digits = 1)

matches <- which(walkTest_labels == walkTest_N_pred_k30)
round((length(matches) / length(walkTest_labels)) * 100, digits = 1)

matches <- which(walkTest_labels == walkTest_Z_pred_k10)
round((length(matches) / length(walkTest_labels)) * 100, digits = 1)

matches <- which(walkTest_labels == walkTest_Z_pred_k30)
round((length(matches) / length(walkTest_labels)) * 100, digits = 1)

# The results indicate that using normalized data is superior to using
# z-scored data.  The difference in accuracy between k = 10 and k = 30
# is minimal, so we hope to find a point of diminishing returns somewhere
# between these two values


# Run a loop that tests a range of values for k to determine best setting
# a table is then printed which shows the results.  In this case it will test
# every odd number between 11 and 29.  The results dataframe that is created
# before the loop is used to store the results of each trial.

results <- data.frame(k = character(0), results = numeric(0))

for (i in 5:14) {
  num_k <- (i * 2 + 1)
  walkTest_pred <- knn(train = normal_train, test = normal_test,
                       cl = walkTrain_labels, k = num_k)
  
  walk_N_matches <- which(walkTest_labels == walkTest_pred)
  
  col1 <- paste("k =", toString(num_k), sep = " ")
  col2 <- round((length(walk_N_matches) / nrow(normal_test) * 100), 
                digits = 1)
  cols <- data.frame(k = col1, results = col2)
  
  results <- rbind(results, cols)
}

results

# The results indicate that the best value for k is 13, with an overall accuracy
# of 64%.  THIS RESULT WILL VARY IF THE SEED VALUE SET EARLIER IN THE CODE IS
# CHANGED.  Depending on seed values used the best results can vary between 13 and 17.
# We will run the algorithm one more time with k=13 to create a vector of
# predictions to use in the next step.

walkTest_N_pred_k13 <- knn(train = normal_train, 
                           test = normal_test,
                           cl = walkTrain_labels,
                           k = 13)

# Since an accuracy of 64% is far better than random guessing, but not good enough to
# use in an application a 'bucket voting' system is implemented.  If it is assumed
# that a sample of observations can be gathered before the classification is made, each
# observation can be predicted and "bucketed" into the class prediction.  After a
# series of observations is made the 'votes' in each 'bucket' can be counted, and the 
# majority class prediction chosen as the class for the observations.

# This code produces a function which creates a barplot out of a randome sample of 
# observations in a given class to show which class has the most 'votes'.  It takes 
# the class number, the desired random sample size, the vector of test labels, 
# and vector of test predictions as parameters.  The bar which represents the correct 
# class will be colored red, and the incorrect classes will be colored blue.  This 
# graphic illustrates that by using this system the correct class can be determined 
# with an extremely high level of accuracy.

create_bplot <- function(class, sample_size, class_data, predict_data){
  class_list <- unique(class_data)
  index <- which(class_data == class)
  sample <- index[sample(1:length(index), sample_size, replace = FALSE)]
  predictions <- predict_data[sample]
  this_class <- ifelse(class_list == class, "red", "blue")
  barplot(table(predictions),
          col = this_class,
          border = "green",
          main = "Test for Classifcation Accuracy",
          ylab = "Count",
          xlab = "Class",
          ylim = c(0, sample_size))
}

# The function is called a couple of times to illustrate how it works.  Simply by
# changing the parameters for class number and sample size the function can be used
# again to test different set.

create_bplot(16, 50, walkTest_labels, walkTest_N_pred_k13)

create_bplot(5, 25, walkTest_labels, walkTest_N_pred_k13)