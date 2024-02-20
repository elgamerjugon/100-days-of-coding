import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from  IPython.display import display, HTML
import matplotlib.pyplot as plt

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
    "avg_cars_at home(approx).1": "avg_cars_at_home"
    }, inplace = True)

df.hist(bins = 50, figsize = (15, 8))
plt.show

hist = go.Histogram(x = df.store_sales, name = "Histogram", histnorm = "probability density")
continuous_variables_df = df[["store_sales", "unit_sales", "total_children", "num_children_at_home", "avg_cars_at_home", "gross_weight", "cost"]]
fig = px.scatter_matrix(
    continuous_variables_df,
    title="Scatter Matrix of Continuous Variables",
    dimensions=["store_sales", "unit_sales", "total_children", "num_children_at_home", "avg_cars_at_home", "gross_weight", "cost"],
    # color="store_sales",  # Customize color based on a variable (optional)
    # symbol="store_sales",  # Customize marker symbol based on a variable (optional)
    labels={col: col.replace("_", " ").title() for col in continuous_variables_df.columns}  # Custom axis labels
)

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=dict(l=50, r=50, b=50, t=50),
)
fig.show()












# Create the scatter matrix plot
fig = px.scatter_matrix(
    continuous_variables_df,
    title="Scatter Matrix of Continuous Variables",
    dimensions=["store_sales", "unit_sales", "total_children", "num_children_at_home", "avg_cars_at_home", "gross_weight", "cost"],
    # color="store_sales",  # Customize color based on a variable (optional)
    symbol="store_sales",  # Customize marker symbol based on a variable (optional)
    labels={col: col.replace("_", " ").title() for col in continuous_variables_df.columns}  # Custom axis labels
)

# Adjust subplot size and layout
fig.update_layout(
    autosize=False,
    width=1000,
    height=1000,
    margin=dict(l=50, r=50, b=50, t=50),
)

# Show the plot
fig.show()