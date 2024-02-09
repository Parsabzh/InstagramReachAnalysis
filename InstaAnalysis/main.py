import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# load dataset
data = pd.read_csv("Instagram.csv", encoding="latin1")
print(data.head())

# check and drop null
print(data.isnull().sum())
data=data.dropna()

