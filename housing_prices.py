import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from IPython.display import display, HTML

df = pd.read_csv("./train.csv")
df.info()
df.shape
df = df.reset_index()

def create_scrollable_table(df, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += '</div>'
    return html

# EDA on numerical features

numerical_features = df.select_dtypes(include = [np.number])
numerical_features.describe()
numerical_features.head()

# Summary statistics for numerical features
summary_stats = numerical_features.describe().T
html_numerical = create_scrollable_table(summary_stats, "numerical_features", "Summary Statistics for numerical featues")
display(HTML(html_numerical))

# Summary statistics for categorical features
categorical_features = df.select_dtypes(include = [object])
cat_summary_stats =  categorical_features.describe().T
html_categorical = create_scrollable_table(cat_summary_stats, "categorical_features", "Summary statistics for categorical features")
display(HTML(html_categorical))

# Counting null values

null_values = df.isnull().sum() / df.shape[0]
html_null_values = create_scrollable_table(null_values.to_frame(), "null_values", "Null values in the dataset")
display(HTML(html_null_values))

# Percentage of null values
null_values_p = (df.isnull().sum() / df.shape[0]) * 100
html_null_values_p = create_scrollable_table(null_values.to_frame(), "null_values", "Null values in the dataset")
display(HTML(html_null_values_p))

# Showing rows with null values 

rows_missing_values = df[df.isnull().any(axis = 1)]
html_rows_missing_values = create_scrollable_table(rows_missing_values, "rows_missing_values", "Rows with missing values")
display(HTML(html_rows_missing_values))

# The dependant variable is the one that we are trying to predict
# IV exploration

# test = go.Histogram(x = df.SalePrice, nbinsx = 50, name = "Histogram", opacity = 0.75, histnorm = "probability density", marker = dict(color = "purple"))
# test_fig = go.Figure(data = test)
# test_fig.update_layout(
#     title="SalePrice Distribution",
#     xaxis_title="SalePrice",
#     yaxis_title="Density",
#     legend_title_text="Fitted Normal Distribution",
#     plot_bgcolor='rgba(32, 32, 32, 1)',
#     paper_bgcolor='rgba(32, 32, 32, 1)',
#     font=dict(color='white')
# )

mu, sigma = stats.norm.fit(df.SalePrice)

hist_data = go.Histogram(x = df.SalePrice, nbinsx = 50, name = "Histogram", opacity = 0.75, histnorm = "probability density", marker = dict(color = "purple"))

x_norm = np.linspace(df.SalePrice.min(), df.SalePrice.max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)

norm_data = go.Scatter(x = x_norm, y = y_norm, mode = "lines", name = f"Normal dist. (μ={mu:.2f}, σ={sigma:.2f})", line=dict(color="green"))
fig = go.Figure(data = [hist_data, norm_data])

fig.update_layout(
    title="SalePrice Distribution",
    xaxis_title="SalePrice",
    yaxis_title="Density",
    legend_title_text="Fitted Normal Distribution",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)

# Q-Q plot

qq_data = stats.probplot(df.SalePrice, dist = "norm")
qq_fig = px.scatter(x = qq_data[0][0], y = qq_data[0][1], labels = {"x": "Theoretical Quantiles", "y": "Ordered Values"}, color_discrete_sequence = ["purple"])
qq_fig.update_layout(
    title="Q-Q plot",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)

# Adding the line that best fits if no skewness

slope, intercept, r_value, p_value, std_err = stats.linregress(qq_data[0][0], qq_data[0][1])
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

line_data = go.Scatter(x = line_x, y = line_y, mode= "lines", name = "Normal Line", line = dict(color = "green"))

qq_fig.add_trace(line_data)

# Distribution of dwelling types and their relation to sale prices

dwelling_types = df.BldgType.value_counts()
dwelling_prices = df.groupby("BldgType")["SalePrice"].mean()
formatted_dwelling_prices = ["$" + f"{value: ,.2f}" for value in dwelling_prices.values]

# Bar charts
fig1 = go.Figure(data=[go.Bar(
    x=dwelling_types.index,
    y=dwelling_types.values,
    marker_color='rgb(76, 175, 80)',
    text=dwelling_types.values,
    textposition='outside',
    width=0.4,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig1.update_layout(
    title='Distribution of Building Types',
    xaxis_title='Building Type',
    yaxis_title='Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig2 = go.Figure(data=[go.Bar(
    x=dwelling_prices.index,
    y=dwelling_prices.values,
    marker_color='rgb(156, 39, 176)',
    text=formatted_dwelling_prices,
    textposition='outside',
    width=0.4,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])

fig2.update_layout(
    title='Average Sale Price by Building Type',
    xaxis_title='Building Type',
    yaxis_title='Price',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

# There are more 1 fam building types than any other types
# The sale price seems to be related with the building type

# Street and alley access types effect on sales price
street_prices = df.groupby("Street")["SalePrice"].mean()
alley_prices = df.groupby("Alley")["SalePrice"].mean()

# What this function does is like an if statement, if index == Pave, then the 
# value is purple, if not the the valie is green
colors_street = np.where(street_prices.index == "Pave", "purple", "green")

fig5 = px.bar(x=street_prices.index, y=street_prices.values, title='Average Sale Price by Street Type',
              template='plotly_dark', text=street_prices.values,
              color=colors_street, color_discrete_sequence=['purple', 'green'])

fig5.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
fig5.update_yaxes(title='Sale Price', tickprefix='$', tickformat=',')
fig5.update_xaxes(title='Street Type')
fig5.update_layout(showlegend=False)

# Alley Prices
colors_alley = np.where(alley_prices.index == 'Pave', 'purple', 'green')
fig6 = px.bar(x=alley_prices.index, y=alley_prices.values, title='Average Sale Price by Alley Type',
              template='plotly_dark', text=alley_prices.values,
              color=colors_alley, color_discrete_sequence=['purple', 'green'])

fig6.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
fig6.update_yaxes(title='Sale Price', tickprefix='$', tickformat=',')
fig6.update_xaxes(title='Alley Type')
fig6.update_layout(showlegend=False)

# Average sale price by propery shape
colors = px.colors.qualitative.Plotly

shape_prices = df.groupby("LotShape")["SalePrice"].mean()
contour_prices = df.groupby("LandContour")["SalePrice"].mean()

# Shape Prices
fig7 = px.bar(x=shape_prices.index, y=shape_prices.values, title='Average Sale Price by Property Shape',
              template='plotly_dark', text=shape_prices.values)

fig7.update_traces(marker_color=colors, texttemplate='$%{text:,.0f}', textposition='outside')
fig7.update_yaxes(title='Sale Price', tickprefix='$', tickformat=',')
fig7.update_xaxes(title='Property Shape')
fig7.update_layout(showlegend=False)

# Contour Prices
fig8 = px.bar(x=contour_prices.index, y=contour_prices.values, title='Average Sale Price by Property Contour',
              template='plotly_dark', text=contour_prices.values)

fig8.update_traces(marker_color=colors, texttemplate='$%{text:,.0f}', textposition='outside')
fig8.update_yaxes(title='Sale Price', tickprefix='$', tickformat=',')
fig8.update_xaxes(title='Property Contour')
fig8.update_layout(showlegend=False)

# It also looks like there's some relation between
# the contour, land shape and the price

# Calculating property age
df["PropertyAge"] = df.YrSold - df.YearBuilt

# Creating a graph to see for correlation between variables
# of propery age and Sale Price

age_price_corr = df.PropertyAge.corr(df["SalePrice"])
age_price_corr

# Visualize the correlation
fig9 = px.scatter(df, x = "PropertyAge", y = "SalePrice",
                  title = "Property Age vs Sale Price", color = "PropertyAge",
                  color_continuous_scale = px.colors.sequential.Purp)

fig9.update_layout(plot_bgcolor = "rgb(30, 30, 30)", paper_bgcolor = "rgb(30, 30, 30)", font = dict(color = "white"))
fig9.show()

# The correlation coeficient is kind of strong, it has a 
# negative effect, while the property age is higher, the price
# tend to be lower

# Calculating the correlation between living area and Sale Price

living_area_corr = df.GrLivArea.corr(df.SalePrice)
living_area_corr

# The coeficient is higher with a positive impact

fig10 = px.scatter(df, x = "GrLivArea", y = "SalePrice",
                  title = "GrLivArea vs Sale Price", color = "GrLivArea",
                  color_continuous_scale = px.colors.sequential.Purp)

fig10.update_layout(plot_bgcolor = "rgb(30, 30, 30)", paper_bgcolor = "rgb(30, 30, 30)", font = dict(color = "white"))
fig10.show()

# Start with boxplots
yearly_avg_sale_price = df.groupby("YrSold")["SalePrice"].mean()

fig13 = px.box(df, x = "YrSold", y = "SalePrice", 
               title = "Sale Price Trend Over the Years",
               points = False, color_discrete_sequence = ["green"])

fig13.add_trace(px.line(x = yearly_avg_sale_price.index, y = yearly_avg_sale_price).data[0])

display(HTML(create_scrollable_table(pd.DataFrame(df[df.YrSold == 2007].SalePrice.sort_values()), "2007-sales", "2007 Sales")))

fig13.update_traces(line = dict(color = "purple", width = 4), selector = dict(type = "scatter", mode = "lines"))

for year, avg_price in yearly_avg_sale_price.items():
    fig13.add_annotation(
        x = year,
        y = avg_price,
        text = f"{avg_price:,.0f}",
        font = dict(color = "white"),
        showarrow = False,
        bgcolor = "rgba(128, 0, 128, 0.6)"
    )

fig13.update_layout(
    plot_bgcolor = "rgb(30, 30, 30)",
    paper_bgcolor = "rgb(30, 30, 30, 30)",
    font = dict(color = "white"),
    xaxis_title = "Year Sold",
    yaxis_title = "Sale Price"
)
# df.info()
# df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
# df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.int32)

# Creating pipelines with sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Transformers
# El problema que tenía era la distribución de los paréntesis en los Pipelines, los steps no los estaba jerarquizando bien
numerical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "mean")),
                                          ("scaler", StandardScaler())
                                          ])

categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                                            ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse = False))
                                            ])

# Update categorical and numerical values
categorical_columns = df.select_dtypes(include = ["object", "category"]).columns
# numerical_columns = df.select_dtypes(include = ["int32", "float32"]).columns
numerical_columns = df.select_dtypes(include = ["int64", "float64"]).columns


numerical_columns = numerical_columns.drop("SalePrice")

# Combine transformers and start preprocessing
preprocessor = ColumnTransformer(
    transformers = [
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)],
        remainder = "passthrough" 
    )

pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor)])

# Apply the pipeline to dataset

X = df.drop("SalePrice", axis = 1)
y = np.log(df["SalePrice"]) 
X_preprocessed = pipeline.fit_transform(X)
# X_preprocessed = np.float32(X_preprocessed)


# np.any(np.isnan(X_preprocessed))
# np.all(np.isfinite(X_preprocessed))
# np.isinf(X_preprocessed) == True
# display(HTML(create_scrollable_table(pd.DataFrame(X_preprocessed), "preprocessed", "X")))
# np.where(np.isnan(X_preprocessed))
# Start impleenting regressors

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size = 0.2, random_state = 42)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_state = 42)
}

# Hyperparameters
param_grids = {
    "LinearRegression": {},
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 30],
        "min_samples_split": [3, 6, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [None, 10, 30],
        "max_depth": [3, 6, 10]
    }
}

# Cross validation
cv = KFold(n_splits = 3, shuffle = True, random_state = 42)

# Training and tuning

grids = {}
for model_name, model in models.items():
    print(f"El modelo es: {model_name}")
    grids[model_name] = GridSearchCV(estimator = model, 
                                     param_grid = param_grids[model_name],
                                     cv = cv,
                                     scoring = "neg_mean_squared_error",
                                     n_jobs = -1,
                                     verbose = 2)
    grids[model_name].fit(X_train, y_train)
    best_params = grids[model_name].best_params_
    best_score = np.sqrt(-1 * grids[model_name].best_score_)

    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best RMSE for {model_name}: {best_score}\n")

# np.where(np.isnan(y))
    
# Train nerual network

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Create MLP Regressor Instance
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(random_state = 42, max_iter = 10000, n_iter_no_change = 3, learning_rate_init = 0.001)

# Parameter grid for tuning
param_grid = {
    "hidden_layer_sizes": [(10,), (10, 10), (10, 10, 10), (25)],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "invscaling", "adaptive"]

}

# CReate GridSearchCV object
grid_search_mlp = GridSearchCV(mlp, param_grid, scoring = "neg_mean_squared_error", cv = 3, n_jobs = -1, verbose = 1)

# Fit the model on the training data
grid_search_mlp.fit(X_train_scaled, y_train)

# Print best parameters found during search
print("Best parameters found:", grid_search_mlp.best_params_ )

# Evaluate the model on the test data
best=score = np.sqrt(-1 * grid_search_mlp.best_score_)
print("Test score:", best_score)

# Principal component analysis
from sklearn.decomposition import PCA

pca = PCA()
x_pca_pre = pca.fit_transform(X_preprocessed)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Choose the number of components based on the explained variance
n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

pca = PCA(n_components = n_components)
pipeline_pca = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("pca", pca)
])

X_pca = pipeline_pca.fit_transform(X)

# Running same models after applying PCA
# Splitting data

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)

# Define the models to work with
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_State = 42)
}

# Define hyperatameters
param_grids = {
    "LinearRegression": {},
    "RandomForest":{
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 30],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 6, 10]
    }
}

# 3 fold cross validation
cv = KFold(n_splits = 3, shuffle = True, random_state = 42)

# Train and tune the models
grids_pca = {}
for model_name, model in models.items():
    grids_pca[model_name] = GridSearchCV(
        estimator = model, 
        param_grid = param_grids[model_name], 
        cv = cv,
        scoring = "neg_mean_squared_error",
        n_jobs = -1, 
        verbose = 2
        ) 
    grids_pca[model_name] .fit(X_train_pca, y_train_pca)
    best_params = grids_pca[model_name].best_params_
    best_score = np.sqrt(-1 * grids_pca[model_name].best_score_)
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best RMSE for {model_name}: {best_score}\n")


# Neural Network
X_train_scaled_pca = X_train_pca.copy()
X_test_scaled_pca = X_test_pca.copy()

# MLP Regressor instance
mlp = MLPRegressor(
    random_state = 42, 
    max_iter = 10000, 
    n_iter_no_change = 3,
    learning_rate_init = 0.001)

# Define the parameter grid for tuning
param_grid = {
    "hidden_layer_sizes": [(10, ), (10, 10), (10, 10, 10), (25)],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001, 0.01, .1, 1],
    "learning_rate": ["constant", "invscaling", "adaptive"]
}

# Create Gridsearch object
grid_search_mlp_pca = GridSearchCV(
    mlp, 
    param_grid, 
    scoring = "neg_mean_squared_error",
    cv = 3, 
    n_jobs = -1,
    verbose = 1)

grid_search_mlp_pca.fit(X_train_scaled_pca, y_train)

print(f"Best parameters found: {grid_search_mlp_pca.best_params_}")

best_score = np.sqrt(-1 * grid_search_mlp_pca.best_score_)
print(f"Best score: {best_score}")

from sklearn.metrics import mean_squared_error
for i in grids.keys():
    print(i + ":", str(np.sqrt(mean_squared_error(grids[i].predict(X_test), y_test))))

for i in grids.keys():
    print(i + ":", str(np.sqrt(mean_squared_error(grids_pca[i].predict(X_test_pca), y_test_pca))))

print(str(np.sqrt(mean_squared_error(grid_search_mlp.predict(X_test_scaled), y_test))))
print(str(np.sqrt(mean_squared_error(grid_search_mlp_pca.predict(X_test_scaled_pca), y_test))))

# Variables exploration for feature engineering
var_explore = df[['Fence','Alley','MiscFeature','PoolQC','FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrType','Electrical','MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType','LotFrontage','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','TotalBsmtSF']]

display(HTML(create_scrollable_table(var_explore, 'var_explore', 'List of Variables to Explore for Feature Engineering')))

from sklearn.preprocessing import FunctionTransformer

# feature engineering functions 
def custom_features(df):
    df_out = df.copy()
    df_out['PropertyAge'] = df_out['YrSold'] - df_out['YearBuilt']
    df_out['TotalSF'] = df_out['TotalBsmtSF'] + df_out['1stFlrSF'] + df_out['2ndFlrSF']
    df_out['TotalBath'] = df_out['FullBath'] + 0.5 * df_out['HalfBath'] + df_out['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df_out['HasRemodeled'] = (df_out['YearRemodAdd'] != df_out['YearBuilt']).astype(object)
    df_out['Has2ndFloor'] = (df_out['2ndFlrSF'] > 0).astype(object)
    df_out['HasGarage'] = (df_out['GarageArea'] > 0).astype(object)
    df_out['YrSold_cat'] = df_out['YrSold'].astype(object)
    df_out['MoSold_cat'] = df_out['MoSold'].astype(object)
    df_out['YearBuilt_cat'] = df_out['YearBuilt'].astype(object)
    df_out['MSSubClass_cat'] = df_out['MSSubClass'].astype(object)
    
    return df_out

feature_engineering_transformer = FunctionTransformer(custom_features)
# Update categorical and numerical variables from the new ones 
new_cols_categorical = pd.Index(['HasRemodeled', 'Has2ndFloor', 'HasGarage'])
new_cols_numeric = pd.Index(['PropertyAge', 'TotalSF', 'TotalBath', 'YrSold_cat', 'MoSold_cat', 'YearBuilt_cat', 'MSSubClass_cat'])

# Update categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.append(new_cols_categorical)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.append(new_cols_numeric)

# Remove target variable from numerical columns
numerical_columns = numerical_columns.drop('SalePrice')

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],remainder = 'passthrough')

# Create a pipeline with the preprocessor
pipeline_fe = Pipeline(steps=[
    ('fe', feature_engineering_transformer),
    ('preprocessor', preprocessor),
    ('pca', pca)])

# Apply the pipeline to your dataset
X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])
X_preprocessed_fe = pipeline_fe.fit_transform(X)

# Train and test our models
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_preprocessed_fe, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Define the hyperparameter grids for each model
param_grids = {
    'LinearRegression': {},
    'RandomForest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 30],
        'min_samples_split': [2, 5, 10],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 6, 10],
    }
}

# 3-fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Train and tune the models
grids_fe = {}
for model_name, model in models.items():
    #print(f'Training and tuning {model_name}...')
    grids_fe[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grids_fe[model_name].fit(X_train_fe, y_train_fe)
    best_params = grids_fe[model_name].best_params_
    best_score = np.sqrt(-1 * grids_fe[model_name].best_score_)
    
    print(f'Best parameters for {model_name}: {best_params}')
    print(f'Best RMSE for {model_name}: {best_score}\n')