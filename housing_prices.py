import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from IPython.display import display, HTML

df = pd.read_csv("./train.csv")
df.info()
df.shape

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
