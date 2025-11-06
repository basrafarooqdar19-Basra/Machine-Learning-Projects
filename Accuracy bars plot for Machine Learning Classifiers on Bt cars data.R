
# Install and load the necessary packages
library(e1071)
library(caret)
library(ggplot2)
library(MASS)
library(rpart)
library(class)

# Load the dataset (assuming it's a dataframe named ship_data)
# For demonstration purposes, let's assume ship_data is your dataframe
# ship_data <- read.csv("your_data.csv")

# Let's use the built-in dataset mtcars for demonstration purposes
data(mtcars)
ship_data <- mtcars
ship_data$class <- ifelse(ship_data$mpg > median(ship_data$mpg), "high", "low")
ship_data$class <- as.factor(ship_data$class)

# Define the number of folds for cross-validation
k <- 5

# Create folds for cross-validation
set.seed(123)
folds <- createFolds(ship_data$class, k = k, list = TRUE, returnTrain = FALSE)

# Define the percentages of features to select
percentages <- seq(0.2, 1, 0.1)

# Define the classifiers
classifiers <- c("svm", "lda", "rpart", "knn")

# Initialize a list to store the accuracy results
accuracy_results <- lapply(classifiers, function(x) matrix(nrow = k, ncol = length(percentages)))

# Perform feature selection and train models for each fold
for (i in 1:k) {
  validation_index <- folds[[i]]
  training_data <- ship_data[-validation_index, ]
  validation_data <- ship_data[validation_index, ]

  # Apply feature selection using t-test on the training data
  p_values <- sapply(1:(ncol(training_data) - 1), function(i) {
    t.test(as.numeric(training_data[, i]) ~ training_data$class)$p.value
  })
  feature_order <- order(p_values)

  # Select features for each percentage and train models
  for (j in 1:length(percentages)) {
    percent <- percentages[j]
    num_features <- round((ncol(training_data) - 1) * percent)
    selected_features <- feature_order[1:num_features]

    # Train models on the selected features
    for (c in 1:length(classifiers)) {
      if (classifiers[c] == "svm") {
        model <- svm(class ~ ., data = training_data[, c(selected_features, ncol(training_data))], kernel = "radial")
      } else if (classifiers[c] == "lda") {
        model <- lda(class ~ ., data = training_data[, c(selected_features, ncol(training_data))])
      } else if (classifiers[c] == "rpart") {
        model <- rpart(class ~ ., data = training_data[, c(selected_features, ncol(training_data))], method = "class")
      } else if (classifiers[c] == "knn") {
        model <- NULL
      }

      # Predict on the validation data
      if (classifiers[c] == "knn") {
        predictions <- knn(training_data[, selected_features], validation_data[, selected_features], training_data$class, k = 5)
      } else {
        predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))])
        if (classifiers[c] == "rpart") {
          predictions <- predictions[, 2]
          predictions <- ifelse(predictions > 0.5, levels(training_data$class)[2], levels(training_data$class)[1])
        } else if (classifiers[c] == "lda") {
          predictions <- predict(model, newdata = validation_data[, c(selected_features, ncol(validation_data))])$class
        }
      }

      # Calculate the accuracy
      accuracy <- sum(predictions == validation_data$class) / nrow(validation_data)

      # Store the accuracy results
      accuracy_results[[c]][i, j] <- accuracy
    }
  }
}

# Calculate the average accuracy for each percentage of features and each classifier
average_accuracy <- lapply(accuracy_results, function(x) apply(x, 2, mean))
print(average_accuracy)

# Calculate the overall average accuracy for each classifier
overall_average_accuracy <- sapply(average_accuracy, mean)

# Create a data frame for plotting
plot_data <- data.frame(
  Classifier = classifiers,
  Average_Accuracy = overall_average_accuracy
)

# Plot the average accuracy results
ggplot(plot_data, aes(x = Classifier, y = Average_Accuracy, fill = Classifier)) +
  geom_col() +
  labs(x = "Classifier", y = "Average Accuracy") +
  theme_classic() +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "none")