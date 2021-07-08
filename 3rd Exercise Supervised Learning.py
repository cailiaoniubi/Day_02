# Jian Hu is writing the code here
# ''' Day3 Exercise for Supervised Machine Learning '''

# import the methods and packages we need
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


# 1. Feature engineering

# a)load the Boston House Price data set
boston = load_boston()
X = boston["data"]
y = boston["target"]


# b) extract polynomial features and interactions up to degree of 2 using PolynomialFeatures, we have 104 features
poly = PolynomialFeatures(2, include_bias=False)
Xpoly = poly.fit_transform(X)
print(Xpoly.shape)
print(np.shape(Xpoly))


# c) Create a pandas DataFrame using the polynomials
df_poly = pd.DataFrame(
    data=poly.fit_transform(X),
    columns=poly.get_feature_names(boston["feature_names"]))

# add the dependent variable to the DataFrame and name the column "y"
df_poly["y"] = y

# save the dataFrame as comma-seperated textile named
df_poly.to_csv("./output/polynomials.csv")


# 2. Regularization


# a) read the data from question "Feature engineering"
df_bostonpoly = pd.read_csv("./output/polynomials.csv", index_col=0)


# b) use column y as target variable and all other columns as predicting variables
X = df_bostonpoly.iloc[:, 0:104]
y = df_bostonpoly["y"]

# splitting them into test and train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# c) learn an ordinary ols model, Ridge Model, and Lasso model

# The R-square score for OLS is 0.7758378393640943
lm = LinearRegression().fit(X_train, y_train)
print(lm.score(X_test, y_test))

# The R-square score for Ridge model is 0.7821576432009781
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
print(ridge.score(X_test, y_test))

# The R-square score for Lasso model is 0.8050991383204581
lasso = Lasso(alpha=0.3).fit(X_train, y_train)
print(lasso.score(X_test, y_test))

# Therefore, Lasso model has highest R-square across the models


# d)Create a DataFrame containing the learned coefficients of all models  and the feature names as Index
df_coefs = pd.DataFrame(lm.coef_)
df_coefs["ridge"] = pd.DataFrame(ridge.coef_)
df_coefs["lasso"] = pd.DataFrame(lasso.coef_)

# renaming both the column and row index names
df_coefs.index = X.columns
df_coefs = df_coefs.rename(columns={0: "OLS"})

# in 37 rows are the Lasso coefficients equal to 0 while the Ridge coefficients are not
len(df_coefs[(df_coefs["lasso"] == 0) & (df_coefs["ridge"] != 0)])


# e) using matplotlib.pyplot, create a horizontal bar plot  of 10*30
plt.rcParams['figure.figsize'] = (10, 30)

# prepare the ticks and all
N = 104  # NUMBER OF FEATURES
ind = np.arange(N)
width = 0.25  # Set the bars width

# bars
plt.barh(ind, df_coefs["OLS"], width, color="r", label='OLS')
plt.barh(ind + width, df_coefs["ridge"], width, color='g', label="ridge")
plt.barh(ind + width*2, df_coefs["lasso"], width, color="b", label="lasso")

# labels, ticks and legend
plt.xlabel("Coefficients")
plt.ylabel("Features")
plt.title("Regression Coefficients for OLS, ridge and lasso regressions")
plt.yticks(ind + width, df_coefs.index)
plt.legend()

# save the figure
plt.savefig("./output/polynomials.pdf")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.close()

# 3. Neural Network Regression
# a) load the diabetes data set
diabetes = load_diabetes()
X = diabetes["data"]
y = diabetes["target"]

# to split the data set as usual
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# b) identity-activation after standard scaling in all nine-parameter

# learn a  Neural Network Regressor with identity activation
# Algorithms to be executed consecutively within each fold
algorithms = [("scaler", StandardScaler()),
              ("nn", MLPRegressor(max_iter=1000, random_state=42, activation='identity', solver='lbfgs'))]
pipe = Pipeline(algorithms)

# Model parameters to be tested one combination at a time
param_grid = {"nn__hidden_layer_sizes": [(10, 10, 10), (42, 42, 42), (110, 110, 110)],
              "nn__alpha": [0.0001, 0.001, 0.01]}

# Train each model
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train, y_train)  # If overall generalization isn't relevant, use X and y

# set output options for print of summary statistics, so that all rows and columns are displayed
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Visualize grid search results
results = pd.DataFrame(grid.cv_results_)
print(results)

# c) The best parameter is {'nn__alpha': 0.001, 'nn__hidden_layer_sizes': (10, 10, 10)},
# they did not perform so well with a score of 0.46723441620830997 less than 0.5
# The model overall generalization score is 0.48490228581338235 which is also not impressive
print(grid.best_params_)
print(grid.best_score_)  # Best parameter combination
print(grid.score(X_test, y_test))  # Overall generalization


# d) plot a heatmap for the first coefficients matrix of the best model, and save it

best = grid.best_estimator_._final_estimator
sns.heatmap(best.coefs_[0], yticklabels=diabetes["feature_names"])
plt.savefig("./output/nn_diabetes_importances.pdf")


# 4. Neural Networks Classification

# a) sklearn.datasets.load_breast_cancer()
cancer = load_breast_cancer()

# as usual, split the data into test adn training set
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)


# b) read up about Area Under the Curve of the Receiver Operating Characteristics, this has been done well.


# c) learn a neural network classifier
# Algorithms to be executed consecutively within each fold
algorithms = [("scaler", MinMaxScaler()),
              ("nn", MLPClassifier(max_iter=1000, random_state=42, solver='lbfgs'))]
pipe = Pipeline(algorithms)

# Model parameters to be tested one combination at a time
param_grid = {"nn__hidden_layer_sizes": [(10, 10, 10), (30, 30, 30)],
              "nn__alpha": [0.0001, 0.001]}

# Train each model
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# set output options for print of summary statistics, so that all rows and columns are displayed
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Visualize grid search results
results = pd.DataFrame(grid.cv_results_)
print(results)

# The best parameter is {'nn__alpha': 0.001, 'nn__hidden_layer_sizes': (30, 30, 30)},
# they did perform so well with a score of 0.9729714019995349
# The model overall generalization score is 0.9850485648804977 which is also not impressive
print(grid.best_params_)
print(grid.best_score_)  # Best parameter combination
print(grid.score(X_test, y_test))  # Overall generalization


# d) Plot the confusion matrix as heatmap for the best model
preds = grid.predict(X_test)
confusion_m = confusion_matrix(y_test, preds)
sns.heatmap(confusion_m, annot=True, fmt='g')

# save the confusion matrix
plt.savefig("./output/nn_breast_confusion.pdf")


