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
