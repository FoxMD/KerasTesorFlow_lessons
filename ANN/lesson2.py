import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/kc_house_data.csv")

# show distribution of price of the houses
# sns.distplot(df["price"])

# show number of bedrooms in our dataset
# sns.countplot(df["bedrooms"])
# plt.show()

# find out correlations between price and other features and explore them
# print(df.corr()["price"].sort_values())
# sns.scatterplot(x="price", y="sqft_living", data=df)
# plt.show()

# see the distribution of prices in latidude and longitude
# plt.figure(figsize=(12, 12))
# sns.scatterplot(x="long", y="lat", data=df, hue="price")
# plt.show()
# this data has still big influence of the very expensive houses
# lets check it and sample out all houses with a price tag over 3M
# find out the top 1%
print(len(df)*0.01)
non_top_1_perc = df.sort_values("price", ascending=False).iloc[216:]

# make the plot for the 99% of the houses
plt.figure(figsize=(12, 12))
sns.scatterplot(x="long", y="lat", data=non_top_1_perc, edgecolor=None, alpha=0.2, palette="RdYlGn", hue="price")
plt.show()

# boxplot on waterfront
sns.boxplot(x="waterfront", y="price", data=df)
plt.show()