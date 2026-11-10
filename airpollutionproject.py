



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




df = pd.read_csv(
    r"C:\Users\Adithya\Downloads\3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv"
)

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nSIZE OF THE DATASET")
print(df.shape)

print("\nINFORMATION OF THE DATA")
df.info()

print("\nDUPLICATE VALUES")
print(df.duplicated().sum())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDESCRIPTION OF THE DATA")
print(df.describe())

print("\nCOUNT OF VALUES")
print(df.count())





# Convert date and time

df["last_update"] = pd.to_datetime(
    df["last_update"],
    format="%d-%m-%Y %H:%M:%S",
    
)

df["hour"] = df["last_update"].dt.hour

df["month"] = df["last_update"].dt.month

df["year"] = df["last_update"].dt.year


df = df.drop_duplicates()





# State-wise average pollution

group_by_state = df.groupby("state")["pollutant_avg"].mean()

print("\nSTATE-WISE AVERAGE POLLUTION")
print(group_by_state)


# City-wise average pollution

group_by_city = df.groupby(
    ["state", "city"]
)["pollutant_avg"].mean()

print("\nCITY-WISE AVERAGE POLLUTION")
print(group_by_city)


# Station-wise average pollution

group_by_station = df.groupby(
    ["city", "station"]
)["pollutant_avg"].mean()

print("\nSTATION-WISE AVERAGE POLLUTION")
print(group_by_station)




pm25_data = df[
    df["pollutant_id"] == "PM2.5"
]["pollutant_avg"].dropna()


q1 = pm25_data.quantile(0.25)

q2 = pm25_data.quantile(0.75)

iqr = q2 - q1

lower = q1 - 1.5 * iqr

upper = q2 + 1.5 * iqr


outliers = pm25_data[
    (pm25_data < lower) |
    (pm25_data > upper)
]


print("\nPM2.5 OUTLIERS")

print(outliers)

print("\nTOTAL OUTLIERS")

print(len(outliers))





station_data = df.pivot_table(
    index=[
        "country",
        "state",
        "city",
        "station",
        "latitude",
        "longitude"
    ],

    columns="pollutant_id",

    values="pollutant_avg",

    aggfunc="mean"
).reset_index()


print("\nSTATION-WISE DATA")

print(station_data.head())

print("\nSTATION DATA SHAPE")

print(station_data.shape)

print("\nMISSING VALUES AFTER PIVOT")

print(station_data.isnull().sum())









# Visualization 3: City-wise PM2.5 Comparison

city_pollution = (
    df[df["pollutant_id"] == "PM2.5"]
    .groupby("city")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(11, 6))

plt.barh(
    city_pollution.index,
    city_pollution.values
)

plt.gca().invert_yaxis()

plt.title("Top 15 Cities by Mean PM2.5")

plt.xlabel("Mean PM2.5")

plt.ylabel("City")

plt.show()







# REGRESSION MODELS
# Predict PM2.5 Concentration


#  LINEAR REGRESSION MODEL
# Predict PM2.5 Concentration


from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# STEP 1: REMOVE MISSING PM2.5 VALUES


reg_data = station_data.dropna(
    subset=["PM2.5"]
).copy()


# STEP 2: SELECT INDEPENDENT VARIABLES (X)


X = reg_data[
    [
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
        "latitude",
        "longitude"
    ]
]


# STEP 3: SELECT DEPENDENT VARIABLE (Y)


y = reg_data["PM2.5"]


# STEP 4: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: HANDLE MISSING VALUES


imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)


# STEP 6: CREATE LINEAR REGRESSION MODEL


lr_model = LinearRegression()

lr_model.fit(X_train, y_train)



# STEP 7: PREDICTION


lr_predictions = lr_model.predict(X_test)



# STEP 8: MODEL EVALUATION
lr_mae = mean_absolute_error(y_test, lr_predictions)

lr_mse = mean_squared_error(y_test, lr_predictions)

lr_rmse = np.sqrt(lr_mse)

lr_r2 = r2_score(y_test, lr_predictions)


print("\nLINEAR REGRESSION RESULTS")

print("MAE:", lr_mae)

print("MSE:", lr_mse)

print("RMSE:", lr_rmse)

print("R2 SCORE:", lr_r2)


# STEP 9: ACTUAL VS PREDICTED VISUALIZATION


plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    lr_predictions,
    alpha=0.6,
    edgecolor="black"
)

min_value = min(y_test.min(), lr_predictions.min())

max_value = max(y_test.max(), lr_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Linear Regression: Actual vs Predicted PM2.5")

plt.xlabel("Actual PM2.5")

plt.ylabel("Predicted PM2.5")

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# STEP 10: RESIDUAL PLOT


residuals = y_test - lr_predictions

plt.figure(figsize=(9, 6))

plt.scatter(
    lr_predictions,
    residuals,
    alpha=0.6,
    edgecolor="black"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.title("Linear Regression Residual Plot")

plt.xlabel("Predicted PM2.5")

plt.ylabel("Residual Errors")

plt.grid(alpha=0.3)

plt.show()




# MODEL 2: DECISION TREE REGRESSOR

# DECISION TREE REGRESSOR
# Predict PM2.5 Concentration


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 1: CREATE DECISION TREE MODEL


dt_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)


# STEP 2: TRAIN THE MODEL


dt_model.fit(X_train, y_train)



# STEP 3: MAKE PREDICTIONS


dt_predictions = dt_model.predict(X_test)


# STEP 4: MODEL EVALUATION


dt_mae = mean_absolute_error(
    y_test,
    dt_predictions
)

dt_mse = mean_squared_error(
    y_test,
    dt_predictions
)

dt_rmse = np.sqrt(dt_mse)

dt_r2 = r2_score(
    y_test,
    dt_predictions
)


print("\n====================================")
print("DECISION TREE REGRESSOR RESULTS")
print("====================================")

print("MAE:", dt_mae)

print("MSE:", dt_mse)

print("RMSE:", dt_rmse)

print("R2 SCORE:", dt_r2)



# VISUALIZATION 1: ACTUAL VS PREDICTED


plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    dt_predictions,
    alpha=0.6,
    edgecolor="black",
    label="Predicted Values"
)

min_value = min(
    y_test.min(),
    dt_predictions.min()
)

max_value = max(
    y_test.max(),
    dt_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title(
    "Decision Tree: Actual vs Predicted PM2.5",
    fontsize=14
)

plt.xlabel("Actual PM2.5 Values")

plt.ylabel("Predicted PM2.5 Values")

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# VISUALIZATION 2: RESIDUAL PLOT

dt_residuals = y_test - dt_predictions

plt.figure(figsize=(9, 6))

plt.scatter(
    dt_predictions,
    dt_residuals,
    alpha=0.6,
    edgecolor="black"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Decision Tree Residual Plot",
    fontsize=14
)

plt.xlabel("Predicted PM2.5 Values")

plt.ylabel("Residual Errors")

plt.grid(alpha=0.3)

plt.show()




# MODEL 3: RANDOM FOREST REGRESSOR


from sklearn.ensemble import RandomForestRegressor

# PART 8: RANDOM FOREST REGRESSOR


from sklearn.ensemble import RandomForestRegressor


# Create Random Forest Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# Train Model

rf_model.fit(X_train, y_train)


# Make Predictions

rf_predictions = rf_model.predict(X_test)


# MODEL EVALUATION


rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(y_test, rf_predictions)


print("\n======================================")
print("RANDOM FOREST REGRESSOR RESULTS")
print("======================================")

print("MAE:", rf_mae)

print("MSE:", rf_mse)

print("RMSE:", rf_rmse)

print("R2 SCORE:", rf_r2)


# VISUALIZATION 1: ACTUAL VS PREDICTED


plt.figure(figsize=(9, 5))

plt.scatter(
    y_test,
    rf_predictions,
    color="darkorange",
    edgecolor="black",
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title("Random Forest - Actual vs Predicted PM2.5")

plt.xlabel("Actual PM2.5")

plt.ylabel("Predicted PM2.5")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()



# VISUALIZATION 2: RESIDUAL PLOT


rf_residuals = y_test - rf_predictions

plt.figure(figsize=(9, 5))

plt.scatter(
    rf_predictions,
    rf_residuals,
    color="purple",
    alpha=0.7,
    edgecolor="black"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.title("Random Forest - Residual Plot")

plt.xlabel("Predicted PM2.5")

plt.ylabel("Residuals")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()






# PART 10: CLASSIFICATION MODELS
# Predict Pollution Severity



# Create pollution categories


def classify_pollution(value):

    if value <= 30:

        return "Good"

    elif value <= 60:

        return "Satisfactory"

    elif value <= 90:

        return "Moderate"

    elif value <= 120:

        return "Poor"

    elif value <= 250:

        return "Very Poor"

    else:

        return "Severe"


# These thresholds assume PM2.5 is measured in µg/m³.
# Verify the measurement units before using them.


station_data["Pollution_Level"] = (
    station_data["PM2.5"].apply(classify_pollution)
)


# Remove missing PM2.5 values

class_data = station_data.dropna(
    subset=["PM2.5"]
).copy()


# Independent variables

X_class = class_data[
    [
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
        "latitude",
        "longitude"
    ]
]


# Dependent variable

y_class = class_data["Pollution_Level"]


print("\nPOLLUTION CATEGORY COUNTS")

print(y_class.value_counts())


# Train Test Split

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)


# Handle Missing Values

class_imputer = SimpleImputer(strategy="median")

X_train_class = class_imputer.fit_transform(X_train_class)

X_test_class = class_imputer.transform(X_test_class)


# MODEL 1: DECISION TREE CLASSIFIER



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


dt_classifier = DecisionTreeClassifier(
    random_state=42
)

dt_classifier.fit(
    X_train_class,
    y_train_class
)

dt_class_predictions = dt_classifier.predict(
    X_test_class
)


print("\nDECISION TREE CLASSIFIER RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test_class, dt_class_predictions)
)

print("\nClassification Report")

print(
    classification_report(
        y_test_class,
        dt_class_predictions,
        zero_division=0
    )
)


# MODEL 2: RANDOM FOREST CLASSIFIER



from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(
    X_train_class,
    y_train_class
)

rf_class_predictions = rf_classifier.predict(
    X_test_class
)


print("\nRANDOM FOREST CLASSIFIER RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test_class, rf_class_predictions)
)

print("\nClassification Report")

print(
    classification_report(
        y_test_class,
        rf_class_predictions,
        zero_division=0
    )
)



# MODEL 3: SUPPORT VECTOR MACHINE (SVM)



from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_class
)

X_test_scaled = scaler.transform(
    X_test_class
)


# Train SVM Model

svm_model = SVC()

svm_model.fit(
    X_train_scaled,
    y_train_class
)


svm_predictions = svm_model.predict(
    X_test_scaled
)


print("\nSVM CLASSIFIER RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test_class, svm_predictions)
)

print("\nClassification Report")

print(
    classification_report(
        y_test_class,
        svm_predictions,
        zero_division=0
    )
)


# PART 11: CLASSIFICATION MODEL COMPARISON



classification_results = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],

    "Accuracy": [
        accuracy_score(y_test_class, dt_class_predictions),
        accuracy_score(y_test_class, rf_class_predictions),
        accuracy_score(y_test_class, svm_predictions)
    ]

})


print("\nCLASSIFICATION MODEL COMPARISON")

print(classification_results)



# PART 12: ACTUAL VS PREDICTED VISUALIZATION



plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    rf_predictions
)

plt.xlabel("Actual PM2.5")

plt.ylabel("Predicted PM2.5")

plt.title("Actual vs Predicted PM2.5 using Random Forest")

plt.show()


#PART 13: CONFUSION MATRIX


labels = rf_classifier.classes_

cm = confusion_matrix(
    y_test_class,
    rf_class_predictions,
    labels=labels
)


plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Random Forest Pollution Classification")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



# PART 14: FEATURE IMPORTANCE



feature_names = [
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
    "NH3",
    "latitude",
    "longitude"
]


importance = rf_model.feature_importances_


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nRANDOM FOREST FEATURE IMPORTANCE")

print(feature_importance)





print("\n===================================")

print("AIR POLLUTION INTELLIGENCE PROJECT")

print("ALL MODELS EXECUTED SUCCESSFULLY")

print("===================================")
