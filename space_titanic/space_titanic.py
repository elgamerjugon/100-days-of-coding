import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

df = pd.read_csv("./space_titanic_train.csv")
df.info()