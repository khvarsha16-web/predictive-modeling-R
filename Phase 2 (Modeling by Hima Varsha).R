##-----------------Predictive-modeling.R -----------
## DATASET : merged_counties_median_income.xslx

library(readxl)
library(dplyr)
install.packages("ggcorrplot")
library(ggcorrplot)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(neuralnet)
#####--------PRECHECKS------------------
#STEP 1: Load the dataset
data <- read_excel("merged_counties_median_income.xlsx")
head(data)
str(data)
summary(data)

#Rename Columns to make modeling easier
data <- data%>%
  rename(
    Geographic_Area = `Geographic Area`,
    State_FIPS = `State FIPS Code`,
    County_FIPS = `County FIPS Code`,
    Unemployment_Rate = `Unemployment Rate (%)`,
    Labor_Force = `Labor Force`
  )

names(data)

#Check missing values
colSums(is.na(data))

#New Added step : Feature engineering
# Step : Feature Engineering (Added for deeper insights)
data <- data %>%
  mutate(
    Employment_Rate = Employed / Labor_Force,
    Unemployment_Share = Unemployed / Labor_Force
  )

summary(data[, c("Employment_Rate", "Unemployment_Share")])


#Step 2: Exploratory Data Analysis (EDA)
# Updated numeric variable set (clean)
numeric_vars <- data %>%
  select(
    AIGE,
    Labor_Force,
    Median_Household_Income,
    Unemployment_Rate,
    Employment_Rate,
    Unemployment_Share
  )

# Correlation matrix
cor_matrix <- cor(numeric_vars)
cor_matrix

#Correlation Heatmap
ggcorrplot(cor_matrix, lab = TRUE)

#Scatter Plots
#For AIGE vs Income
ggplot(data, aes(x = Median_Household_Income, y = AIGE)) + 
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se= FALSE, color = "blue") +
  theme_minimal()

#For AIGE vs Unemployement Rate
ggplot(data, aes(x = Unemployment_Rate, y = AIGE)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  theme_minimal()

#For AIGE vs Employment Rate
ggplot(data, aes(x = Employment_Rate, y = AIGE)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "purple") +
  theme_minimal()

#For AIGE vs Unemployment Share
ggplot(data, aes(x = Unemployment_Share, y = AIGE)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "darkgreen") +
  theme_minimal()


#######-----------PHASE 2 MODELING--------
#####Step 3: BUILD THE REGRESSION MODEL
# Step 3a : Selecting modeling variables
model_data <- data %>%
  select(AIGE, Median_Household_Income, Unemployment_Rate, Labor_Force)


# Step 3b : Fit the Linear regression model
reg_model <- lm(AIGE ~ Median_Household_Income + Unemployment_Rate + Labor_Force,
                data = model_data)
summary(reg_model)

#Step 3c : Model selection (Backward, Forward, Stepwise)
#Backward Elimination
backward_model <- step(reg_model, direction = "backward", trace = FALSE)
summary(backward_model)

#Forward Selection
null_model <- lm(AIGE ~ 1, data = model_data)
forward_model <- step(
  null_model,
  direction = "forward",
  scope = ~ Median_Household_Income + Unemployment_Rate + Labor_Force,
  trace = FALSE
)
summary(forward_model)

#Stepwise (Both Directions)
stepwise_model <- step(
  null_model,
  scope = ~Median_Household_Income + Unemployment_Rate + Labor_Force,
  trace = FALSE
)
summary(stepwise_model)


#Step 3d : Train / Test Split + RMSE
set.seed(123)

train_index <- createDataPartition(model_data$AIGE, p= 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

#Step 3e : Train the final regression model on training data : Using the selected variable labor_force
final_reg <- lm(AIGE ~ Labor_Force, data = train_data)
summary(final_reg)

#Regression diagnostic plots
par(mfrow = c(2,2))
plot(backward_model)
par(mfrow = c(1, 1)

#Step 3f : Predict on test data

pred <- predict(final_reg, newdata = test_data)
actual <- test_data$AIGE

#Step 3g : Compute RMSE, Rsquared, MAE 
core_metrics <- postResample(pred = pred, obs = actual)
core_metrics

#Step 3h : Extra Metrics (ME & MAPE)
ME <- mean(actual - pred)
MAPE <- mean(abs((actual - pred) / actual)) * 100

performance <- data.frame(
  RMSE = core_metrics["RMSE"],
  R2 = core_metrics["Rsquared"],
  MAE = core_metrics["MAE"],
  ME = ME,
  MAPE = MAPE
)
performance



#####Step 4: DECISION TREE MODEL
#Step 4a : Fitting the Tree Model
tree_data <- model_data
tree_model <- rpart(
  AIGE ~ Median_Household_Income + Unemployment_Rate + Labor_Force,
  data = tree_data,
  method = "anova"
)

tree_model

#Step 4b : Plotting the tree 
rpart.plot(tree_model, type = 2, extra = 101, fallen.leaves = TRUE,
           box.palette = "GnBu", main = "Decision Tree for AIGE")

#Step 4c : Variable importance
vip <- tree_model$variable.importance
barplot(vip, main = "Variable Importance (Decision Tree)",
        col = "steelblue", las = 2)

#Step 4d : Predict on test data
tree_pred <- predict(tree_model, newdata = test_data)

#Step 4e : Compute performance metrics
tree_core <- postResample(pred = tree_pred, obs = actual)
tree_core

#Step 4f : Extra metrics
tree_ME <- mean(actual - tree_pred)
tree_MAPE <- mean(abs((actual - tree_pred) / actual)) * 100

tree_performance <- data.frame(
  RMSE = tree_core["RMSE"],
  R2 = tree_core["Rsquared"],
  MAE = tree_core["MAE"],
  ME = tree_ME,
  MAPE = tree_MAPE
)

tree_performance

#####Step 5: NEURAL NETWORK MODEL

#Step 5a : Normalize(scale) the variables
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

nn_data <- model_data %>%
  mutate(
    Median_Household_Income = normalize(Median_Household_Income),
    Unemployment_Rate = normalize(Unemployment_Rate),
    Labor_Force = normalize(Labor_Force)
  )

head(nn_data)

#Step 5b : Train the neural network
nn_train <- nn_data[train_index, ]
nn_test <- nn_data[-train_index, ]

#Step 5c : Neural network with 1 hidden layer (3 neurons)
set.seed(123)

nn_model_1 <- neuralnet(
  AIGE ~ Median_Household_Income + Unemployment_Rate + Labor_Force,
  data = nn_train,
  hidden = 3,
  linear.output = TRUE
)

nn_model_1

#Plot the neural network
plot(nn_model_1, rep = "best")

# Predict on test data (Neural Net with 1 hidden layer)
nn_pred_raw <- compute(
  nn_model_1,
  nn_test[, c("Median_Household_Income", "Unemployment_Rate", "Labor_Force")]
)

nn_pred <- as.vector(nn_pred_raw$net.result)
actual_nn <- nn_test$AIGE   # AIGE was not normalized

# Core metrics (RMSE, R², MAE)
nn_core <- postResample(pred = nn_pred, obs = actual_nn)
nn_core

# Extra metrics (ME, MAPE)
nn_ME <- mean(actual_nn - nn_pred)
nn_MAPE <- mean(abs((actual_nn - nn_pred) / actual_nn)) * 100

nn_performance <- data.frame(
  RMSE = nn_core["RMSE"],
  R2   = nn_core["Rsquared"],
  MAE  = nn_core["MAE"],
  ME   = nn_ME,
  MAPE = nn_MAPE
)

nn_performance


# Step 5d: Neural Network with 2 Hidden Layers 
set.seed(123)
nn_model2 <- neuralnet(
  AIGE ~ Median_Household_Income + Unemployment_Rate + Labor_Force,
  data = nn_train,
  hidden = c(2,1),          # lighter 2-layer network
  linear.output = TRUE,
  stepmax = 2e5
)


# Check basic model info 
nn_model2

#Metrics
nn_pred_raw2 <- compute(
  nn_model2,
  nn_test[, c("Median_Household_Income", "Unemployment_Rate", "Labor_Force")]
)
nn_pred2 <- as.vector(nn_pred_raw2$net.result)

nn_core2 <- postResample(pred = nn_pred2, obs = nn_test$AIGE)
nn_core2

nn_ME2   <- mean(nn_test$AIGE - nn_pred2)
nn_MAPE2 <- mean(abs((nn_test$AIGE - nn_pred2) / nn_test$AIGE)) * 100

nn_performance2 <- data.frame(
  RMSE = nn_core2["RMSE"],
  R2   = nn_core2["Rsquared"],
  MAE  = nn_core2["MAE"],
  ME   = nn_ME2,
  MAPE = nn_MAPE2
)
nn_performance2

#######COMPARISON TABLE
# Model Comparison Table
model_compare <- data.frame(
  Model = c("Regression",
            "Decision Tree",
            "Neural Net (1 layer)",
            "Neural Net (2 layers)"),
  
  RMSE = c(1.2886,
           1.2681,
           1.2570,
           1.2638),
  
  MAE = c(0.9638,
          0.9309,
          0.8973,
          0.9112),
  
  R2 = c(0.0609,
         0.0791,
         0.0980,
         0.0881),
  
  ME = c(0.0221,
         0.0260,
         0.0353,
         0.0293)
)

print(model_compare)

write.csv(model_compare,
          "model_compare.csv",
          row.names = FALSE)

