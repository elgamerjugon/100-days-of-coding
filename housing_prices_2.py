import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
%matplotlib inline


def create_scrollable_table(df, table_id, title):
    html = f"<h3>{title}</h3>"
    html += f"<div id='{table_id}' style='height:200px'; overflow='auto'>"
    html += df.to_html()
    html += "</div>"
    return html

df = pd.read_csv("./train.csv")

df.info()
display(HTML(create_scrollable_table(pd.DataFrame((df.isna().sum() / df.shape[0]) * 100), "nulls_df", "Nulls percentage")))

df_drop = df.drop(columns = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"])

# As there are some columns with a lot of missing values, I will try 2 approaches, 1 dropping the columns and 2 filling the missing values and see which
# model works better

numerical_columns = df_drop.select_dtypes(include = np.number).columns
categorical_columns = df_drop.select_dtypes(include = ["object", "category"]).columns

# First step is to analyze the dependent variable
df_drop["SalePrice"].describe()

# we can see that we have an outlier
# Also, the mean and the 50th quantile does not match, the mean is being affected by outliers

sns.distplot(df_drop.SalePrice)

# We can see that it has some skewness to the right
print(f"Skewness = {df_drop.SalePrice.skew()}")
print(f"Kurtosis = {df_drop.SalePrice.kurtosis()}")

# The value of skewness is telling us that there is a positive skewness as it's > 0
# Positive kurtosis indicates heavy tailed skewness, negative kurtosis indicates light tailed skewness.
# The expected value of kurtosis for normal distribution is 3, if it's more than 3 we expect to see a very narrow distribution with high peak.

# Relationship with other variables, correlation

corrmat = df_drop.corr()
plt.Figure(figsize = (20, 15))
sns.heatmap(corrmat, vmax=.8, square=True, cmap = "coolwarm", linewidth = 0.5, xticklabels=True, yticklabels=True)
plt.title("Corr Heatmap")
plt.show()

# Important correlations
# (SalePrice, OverallQaul), (GarageYrBlt, YearBuilt), (TotRmsAbvGrd, GrLivArea), (GrLivArea, SalePrice), (1stFlrSF, TotalBsmtSF)
# (GarageArea, GarageCars), (SalePrice, GarageCars), (SalePrice, GarageArea)

# Zooming the correlation Heatmap
k = 10
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(df_drop[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(
    cm, 
    cbar = True, 
    annot = True, 
    square = True, 
    fmt = ".2f", 
    annot_kws = {"size": 10}, 
    yticklabels = cols.values, 
    xticklabels = cols.values,
    cmap = "coolwarm")
plt.show()

# Now let's explore SalePric

sns.set()
cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt"]
sns.pairplot(df_drop[cols], size = 3)

# Preprocessing, filling null values and standarizing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numerical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps = [
    ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse = False)),
    ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing"))
])

# numerical_columns = numerical_columns.drop("SalePrice")

preprocessor = ColumnTransformer(
    transformers = [
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ],
    remainder = "passthrough"
)

# Apply preprocessor to dataframe

pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor)
])

X = df_drop.drop(columns = "SalePrice")
# y = np.log(df_drop.SalePrice)
X_preprocessed = pipeline.fit_transform(X)

df_drop[categorical_columns].info()