# Install and load the necessary packages
library(e1071)
library(caret)
library(ggplot2)
library(MASS)
library(rpart)
library(class)
library(nnet)
library(mlbench)

# Let's use the built-in dataset breast cancer for demonstration purposes
data(BreastCancer)
my_data <- BreastCancer
print(my_data)
summary(my_data)
my_data$Cl.thickness <- as.numeric(as.character(my_data$Cl.thickness))
my_data$Class <- ifelse(my_data$Cl.thickness > median(my_data$Cl.thickness, na.rm = TRUE), "malignant", "benign")
my_data$Class <- as.factor(my_data$Class)

# Convert columns to numeric
for (i in 2:(ncol(my_data) - 1)) {
  if (is.factor(my_data[, i])) {
    my_data[, i] <- as.numeric(as.character(my_data[, i]))
  }
}

print(my_data$Class)

# Define the number of folds for cross-validation
k <- 5

# Create folds for cross-validation
set.seed(123)
folds <- createFolds(my_data$Class, k = k, list = TRUE, returnTrain = FALSE)

# Define the percentages of features to select
percentages <- seq(0.2, 1, 0.1)

# Define the classifiers
classifiers <- c("svm", "lda", "rpart", "knn", "mlp")

# Initialize lists to store the accuracy, precision, recall and f1 results
accuracy_results <- lapply(classifiers, function(x) matrix(nrow = k, ncol = length(percentages)))
precision_results <- lapply(classifiers, function(x) matrix(nrow = k, ncol = length(percentages)))
recall_results <- lapply(classifiers, function(x) matrix(nrow = k, ncol = length(percentages)))
f1_results <- lapply(classifiers, function(x) matrix(nrow = k, ncol = length(percentages)))

# Perform feature selection and train models for each fold
for (i in 1:k) {
  validation_index <- folds[[i]]
  training_data <- my_data[-validation_index, ]
  validation_data <- my_data[validation_index, ]
  
  # Apply feature selection using t-test on the training data
  p_values <- sapply(2:(ncol(training_data) - 1), function(i) {
    t.test(as.numeric(training_data[, i]) ~ training_data$Class)$p.value
  })
  feature_order <- order(p_values)
  feature_order <- feature_order + 1
  
  # Select features for each percentage and train models
  for (j in 1:length(percentages)) {
    percent <- percentages[j]
    num_features <- round((ncol(training_data) - 2) * percent)
    if (num_features == 0) {
      next
    }
    selected_features <- feature_order[1:num_features]
    
    # Train models on the selected features
    for (c in 1:length(classifiers)) {
      training_data <- training_data[complete.cases(training_data), ]
      if (classifiers[c] == "svm") {
        model <- svm(Class ~ ., data = training_data[, c(selected_features, ncol(training_data))], kernel = "radial")
      } else if (classifiers[c] == "lda") {
        model <- lda(Class ~ ., data = training_data[, c(selected_features, ncol(training_data))])
      } else if (classifiers[c] == "rpart") {
        model <- rpart(Class ~ ., data = training_data[, c(selected_features, ncol(training_data))], method = "class")
      } else if (classifiers[c] == "knn") {
        model <- NULL
      } else if (classifiers[c] == "mlp") {
        model <- nnet(Class ~ ., data = training_data[, c(selected_features, ncol(training_data))], size = 10)
      }
      validation_data <- validation_data[complete.cases(validation_data), ]
      
      # Predict on the validation data
      if (classifiers[c] == "knn") {
        predictions <- knn(training_data[, selected_features], validation_data[, selected_features], training_data$Class, k = 5)
      } else {
        if (classifiers[c] == "rpart") {
          predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))], type = "class")
        } else if (classifiers[c] == "lda") {
          predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))])$class
        } else if (classifiers[c] == "svm") {
          predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))])
        } else if (classifiers[c] == "mlp") {
          predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))], type = "class")
        }
      }
      
      # Calculate accuracy, precision, recall, and F1 score
      min_length <- min(length(predictions), length(validation_data$Class))
      predictions <- predictions[1:min_length]
      validation_data$Class <- validation_data$Class[1:min_length]
      predictions <- factor(predictions, levels = levels(validation_data$Class))
      tp <- sum((predictions == "malignant") & (validation_data$Class == "malignant"))
      tn <- sum((predictions == "benign") & (validation_data$Class == "benign"))
      fp <- sum((predictions == "malignant") & (validation_data$Class == "benign"))
      fn <- sum((predictions == "benign") & (validation_data$Class == "malignant"))
      accuracy <- (tp + tn) / (tp + tn + fp + fn)
      precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
      recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
      f1 <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
      accuracy_results[[c]][i, j] <- accuracy
      precision_results[[c]][i, j] <- precision
      recall_results[[c]][i, j] <- recall
      f1_results[[c]][i, j] <- f1
    }
  }
}

# Calculate the average accuracy, precision, recall and F1 for each percentage of features and each classifier
average_accuracy <- lapply(accuracy_results, function(x) apply(x, 2, mean))
average_precision <- lapply(precision_results, function(x) apply(x, 2, mean, na.rm = TRUE))
average_recall <- lapply(recall_results, function(x) apply(x, 2, mean, na.rm = TRUE))
average_f1 <- lapply(f1_results, function(x) apply(x, 2, mean, na.rm = TRUE))


#Bar plots for Accuracy, Precision, Recall, F1 Score
# Replace missing values with 0
results$F1[is.na(results$F1)] <- 0
# Create bar plots for each metric
# Accuracy
dev.new()
ggplot(results, aes(x = Classifier, y = Accuracy)) + 
  geom_bar(stat = "identity") + 
  labs(title = "Accuracy", x = "Classifier", y = "Accuracy") +
  theme_classic()

# Precision
dev.new()
ggplot(results, aes(x = Classifier, y = Precision)) + 
  geom_bar(stat = "identity") + 
  labs(title = "Precision", x = "Classifier", y = "Precision") +
  theme_classic()

# Recall
dev.new()
ggplot(results, aes(x = Classifier, y = Recall)) + 
  geom_bar(stat = "identity") + 
  labs(title = "Recall", x = "Classifier", y = "Recall") +
  theme_classic()

# F1 Score
dev.new()
ggplot(results, aes(x = Classifier, y = F1)) + 
  geom_bar(stat = "identity") + 
  labs(title = "F1 Score", x = "Classifier", y = "F1 Score") +
  theme_classic()

# Calculate the average accuracy, precision, recall and F1 for each percentage of features and each classifier
average_accuracy <- lapply(accuracy_results, function(x) apply(x, 2, mean))
average_precision <- lapply(precision_results, function(x) apply(x, 2, mean, na.rm = TRUE))
average_recall <- lapply(recall_results, function(x) apply(x, 2, mean, na.rm = TRUE))
average_f1 <- lapply(f1_results, function(x) apply(x, 2, mean, na.rm = TRUE))

# Accuracy vS Error rate bar plot

# Calculate error rate
results$Error_Rate = 1 - results$Accuracy

# Replace missing values with 0
results$F1[is.na(results$F1)] <- 0

# Plot accuracy vs error rate
dev.new()
ggplot(results, aes(x = Classifier)) + 
  geom_col(aes(y = Accuracy, fill = "Accuracy"), position = position_dodge()) + 
  geom_col(aes(y = Error_Rate, fill = "Error Rate"), position = position_dodge()) + 
  scale_fill_manual(name = "Metric", values = c("Accuracy" = "blue", "Error Rate" = "red")) + 
  labs(title = "Accuracy vs Error Rate", x = "Classifier", y = "Value") +
  theme_classic() +
  theme(legend.position = "bottom")


# Accuracy Vs Error rate  curve Plots
# Calculate average accuracy and error rate for each fold and each classifier
average_accuracy <- lapply(accuracy_results, function(x) apply(x, 1, mean, na.rm = TRUE))
average_error_rate <- lapply(accuracy_results, function(x) apply(1 - x, 1, mean, na.rm = TRUE))

# Plot accuracy vs error rate for every classifier separately
for (i in 1:length(classifiers)) {
  dev.new()
  plot(average_accuracy[[i]], type = "l", col = "blue", lwd = 3,
       ylim = c(0, 2), 
       xlab = "CV Folds", ylab = "Accuracy Vs Error rate", 
       main = paste("", classifiers[i]))
  lines(average_error_rate[[i]], type = "l", col = "red", lty = 2, lwd = 3)
  legend("topright", c("Accuracy", "Error Rate"), col = c("blue", "red"), lty = c(1, 2), lwd = 3, horiz = TRUE)
}
