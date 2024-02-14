import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from  IPython.display import display, HTML

def create_scrollable_table(df, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += '</div>'
    return html

df = pd.read_csv("./train.csv")
df.head()

# EDA
df.describe().T

# There might be outliers on store sales
# There are some binary columns, I have to check that
# No null values

df.info()
df.recyclable_package.value_counts()
df.low_fat.value_counts()
df.coffee_bar.value_counts()
df.video_store.value_counts()
df.salad_bar.value_counts()
df.prepared_food.value_counts()
df.florist.value_counts()

# All of the above columns are binary
df.isna().sum()
# As expected, no null values on the df

# Let's make some graphs
df.columns
df.rename(columns = {
    "store_sales(in millions)": "store_sales", 
    "unit_sales(in millions)": "unit_sales", 
    "avg_cars_at_home(aprox.)1": "avg_cars_at_home"
    }, inplace = True)

