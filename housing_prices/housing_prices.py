# Basic libraries
import pandas as pd
import numpy as np
from IPython.display import display, HTML

# Graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Data split and model validation
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Imputer
from sklearn.impute import SimpleImputer

# Pipeline  
from sklearn.pipeline import Pipeline, make_pipeline

# Column transformer
from sklearn.compose import ColumnTransformer, make_column_transformer

# metrics
from sklearn.metrics import mean_squared_error

def create_scrollable_table(df):

    # Set the option to display all rows
    pd.set_option("display.max_rows", None)

    # Generate an HTML table from the dataframe
    html_table = df.to_html(classes="scrollable-table")

    # Create a scrollable div and embed the table
    html_code = f"""
    <div style="height: 300px; overflow: auto;">
        {html_table}
    </div>
    """

    # Display the HTML code
    display(HTML(html_code))


df = pd.read_csv("./train.csv")
df.info()
df.describe().T

# First observations
# MasVnrArea is affected by outliers, mean doesn't match with 50% percentile, min is 0, max = 1600
# BsmtFinSF2 may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 1470
# 2ndFlrSF may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 2065
# LowQualFinSF may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 572
# WoodDeckSF may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 857
# OpenPorch may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 547
# EnclosedPorch may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 552
# 3SsnPorch may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 508
# ScreenPorch may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 480
# PoolArea may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 738
# MiscVal may be affected by outliers, mean doesn't match with 50% percentile, min = 0, max = 15,500

df_nulls_percentage = (df.isna().sum()/df.shape[0]) * 100
create_scrollable_table(pd.DataFrame(df_nulls_percentage))

sns.pairplot(df)
num_cols = df.select_dtypes(["int64", "float64"]).columns
cat_cols = df.select_dtypes(["object"]).columns

# Analyze the dependent variable
sns.boxplot(df.SalePrice)

def extract_outliers_boxplot(array):
    iqr_q1 = np.quantile(array, 0.25)
    iqr_q3 = np.quantile(array, 0.75)

    iqr = iqr_q3 - iqr_q1

    # finding upper and lower whiskers
    upper_bound = iqr_q3 + (1.5*iqr)
    lower_bound = iqr_q1 - (1.5*iqr)
    outliers = array[(array <= lower_bound) | (array > upper_bound)]
    return outliers

outliers = extract_outliers_boxplot(df.SalePrice)
len(extract_outliers_boxplot(df.SalePrice))

create_scrollable_table(df[df.SalePrice.isin(outliers)])
