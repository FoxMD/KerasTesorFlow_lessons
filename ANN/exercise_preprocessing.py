import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../DATA/lending_club_info.csv", index_col="LoanStatNew")


def feat_info(col_name):
    print(data.loc[col_name]["Description"])


df = pd.read_csv("../DATA/lending_club_loan_two.csv")
df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})

# find out what data is missing
# length of the data frame and missing in percent
# print(df.isnull().sum() / len(df) * 100)
# mort_acc is 10% missing - problem, emp_title, length are interesting - find out how many unique items are there
# print(df["emp_title"].nunique())
# too much unique titles, not worth categorise like more or less paid jobs
df = df.drop("emp_title", axis=1)
# length of job
# print(sorted(df["emp_length"].dropna().unique()))
emp_length_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years',
                    '9 years', '10+ years']
# sns.countplot(x="emp_length", data=df, order=emp_length_order)
# nice, now show their loan status
# sns.countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
# ok find percentage of people per employment category
emp_co = df[df["loan_status"] == "Charged Off"].groupby("emp_length").count()["loan_status"]
emp_fp = df[df["loan_status"] == "Fully Paid"].groupby("emp_length").count()["loan_status"]
emp_len = emp_co/(emp_co + emp_fp)
df = df.drop("emp_length", axis=1)
# it is the same about 20% people who doesnt paid off
# there are still some data missing
# find if there is some data which repeats
# title is a string to purpose, drop it
df = df.drop("title", axis=1)
# now add missing data (like mortgage - there is almost 10% missing)
# print(df.corr()["mort_acc"].sort_values())
# there is some data correlating, not strong so it isn't a eqv. of it, but it correlates total_acc 38%
# use it to fill na
t_acc_mean = df.groupby("total_acc").mean()["mort_acc"]


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return t_acc_mean[total_acc]
    else:
        return mort_acc


df["mort_acc"] = df.apply(lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1)

df = df.dropna()

# term is a binary column with string, 36 or 60 months, so convert to integer, map it to 36, 60
df["term"] = df["term"].apply(lambda term: int(term[:3]))

# grade feature can be dropped because sub_grade is better
df = df.drop("grade", axis=1)

# convert sub grade to dummy variables
dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
# drop first is because A/B/C = A or B = 0 then C = 1
df = pd.concat([df.drop("sub_grade", axis=1), dummies], axis=1)
# verification_status, application_type, initial_list_status, purpose the same
dummies = pd.get_dummies(df[["verification_status", "application_type", "initial_list_status", "purpose"]],
                         drop_first=True)
df = pd.concat([df.drop(["verification_status", "application_type", "initial_list_status", "purpose"], axis=1),
                dummies], axis=1)

# home ownership mort, rent, own the rest isn't interesting because less than 1%
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df = pd.concat([df.drop("home_ownership", axis=1), dummies], axis=1)

# address column - zip code is interesting
df["zip_code"] = df["address"].apply(lambda address:address[-5:])
# makes 10 zip codes
dummies = pd.get_dummies(df["zip_code"], drop_first=True)
df = pd.concat([df.drop("zip_code", axis=1), dummies], axis=1)
df = df.drop("address", axis=1)

# issue_d the month issued loan funded - this is a data leakage because you don't know this before it is done
df = df.drop("issue_d", axis=1)

# earliest_cr_line is mon-YEAR so change
df["earliest_cr_line"] = df["earliest_cr_line"].apply(lambda date: int(date[-4:]))

# save the data
df.to_csv("../DATA/myLC.csv")

