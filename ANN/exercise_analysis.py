import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../DATA/lending_club_info.csv", index_col="LoanStatNew")


def feat_info(col_name):
    print(data.loc[col_name]["Description"])


df = pd.read_csv("../DATA/lending_club_loan_two.csv")
# print(df.info())
# from the data i can see, there is some data missing, also i can see we have 27 columns and almost 400k entries
# for clasification it is goot to begin with count plot
# sns.countplot(x="loan_status", data=df)
# make a histogram
# sns.distplot(df["loan_amnt"], kde=False, bins=40)
# loo for correlation between the data
# print(df.corr())
# plt sns.heatmap(df.corr(), annot=True, cmap="viridis")
# check if the feature with top (installment) is just another representation of other data (data leakage)
# sns.scatterplot(x="installment",y="loan_amnt", data=df)
# installment is a direct calculation via formula from loan_amnt
# find out if there is loan status dependency on loan amount
# sns.boxplot(x="loan_status", y="loan_amnt", data=df)
# they are almost the same, show numbers
# print(df.groupby("loan_status")["loan_amnt"].describe())
# there is a small difference for th anmnounts
# show categories
# sns.countplot(x="grade", data=df, hue="loan_status")
# show subcategories
# sns.countplot(x="sub_grade", data=df, order=sorted(df["sub_grade"].unique()), hue="loan_status")
# f and g are not paid less often
# filter the F and Gs
# f_and_g = df[(df["grade"] == "G") | (df["grade"] == "F")]
# sns.countplot(x="sub_grade", data=f_and_g, order=sorted(f_and_g["sub_grade"].unique()), hue="loan_status")
df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
# print(df[["loan_repaid", "loan_status"]])
# correlate new repaid data
df.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")
plt.show()

