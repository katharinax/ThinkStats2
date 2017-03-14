#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:13:53 2017

@author: katharina
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import thinkstats2 as ts
import nsfg
import brfss

def ft_inch_to_cm(ft, inch):
    cm = (ft * 12 + inch) * 2.54
    return cm

def stdize(value, mu, sigma):
    z = (value - mu) / sigma
    return z

preg = nsfg.ReadFemPreg()
resp = nsfg.ReadFemResp()
bs = brfss.ReadBrfss()

# Q1. Think Stats Chapter 2 Exercise 4 (effect size of Cohen's d)
live = preg[preg["outcome"] == 1]
first_wt = live.loc[preg["birthord"] == 1, "totalwgt_lb"]
other_wt = live.loc[preg["birthord"] != 1, "totalwgt_lb"]
ts.CohenEffectSize(first_wt, other_wt)

# Q2. Think Stats Chapter 3 Exercise 1 (actual vs. biased)
d = np.diff(np.unique(resp["numkdhh"])).min()
left_of_first_bin = resp["numkdhh"].min() - float(d)/2
right_of_last_bin = resp["numkdhh"].max() + float(d)/2
plt.clf()
plt.hist(resp["numkdhh"], 
         bins = np.arange(left_of_first_bin, right_of_last_bin + d, d), 
         histtype = "step", 
         normed = True,
         label = "Actual")
plt.hist(resp["numkdhh"], 
         bins = np.arange(left_of_first_bin, right_of_last_bin + d, d), 
         histtype = "step", 
         weights = resp["numkdhh"],
         normed = True,
         label = "Biased")
plt.ylabel("Density")
plt.xlabel("# of children")
plt.title("Distribution of number of children under 18 per HH")
plt.legend()
plt.show()

np.average(resp["numkdhh"])
np.average(resp["numkdhh"], weights = resp["numkdhh"])

# Q3. Think Stats Chapter 4 Exercise 2 (random distribution)
rand = np.random.random(1000)
weights = np.ones_like(rand)/1000
plt.clf()
plt.hist(rand, 
         bins = 100,
         histtype = "step",
         weights = weights,
         linewidth = 2,
         label = "PMF")
plt.hist(rand,
         bins = 100,
         histtype = "step",
         cumulative = True,
         weights = weights,
         label = "CDF")
plt.ylabel("Density")
plt.title("Uniform distribution\n(n=1000)")
plt.legend(loc = "upper left")
plt.show()

# Q4. Think Stats Chapter 5 Exercise 1 (normal distribution of blue men)
mu = 178
sigma = 7.7
min_cm = ft_inch_to_cm(5, 10)
min_cm_z = stdize(min_cm, mu, sigma)
max_cm = ft_inch_to_cm(6, 1)
max_cm_z = stdize(max_cm, mu, sigma)   
sp.stats.norm.cdf(max_cm_z) - sp.stats.norm.cdf(min_cm_z)

# Q5. Bayesian (Elvis Presley twin)
((1/300) * (1/2)) / ((1/300) * (1/2) + (1/125) * (1/4))

# Q7. Think Stats Chapter 7 Exercise 1 (correlation of weight vs. age)

## Code by http://markthegraph.blogspot.com/2015/05/using-python-statsmodels-for-ols-linear.html
no_na = live[["agepreg", "totalwgt_lb"]].dropna()
x = no_na["agepreg"]
x2 = sm.add_constant(x)
x_std = (x - x.mean()) / x.std()
x2_std = sm.add_constant(x2)
y = no_na["totalwgt_lb"]
y_std = (y - y.mean()) / y.std()

fitted = sm.OLS(y, x2).fit()
slope = fitted.params.loc["agepreg"]
x_pred = np.linspace(x.min(), x.max(), 50)
x_pred2 = sm.add_constant(x_pred)
y_pred = fitted.predict(x_pred2)
y_hat = fitted.predict(x2)

## CI
y_err = y - y_hat
mean_x = x.T[1].mean()
n = len(x)
dof = n - fitted.df_model - 1
t = stats.t.ppf(1-0.025, df=dof)
s_err = np.sum(np.power(y_err, 2))
conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2) / ((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))
upper = y_pred + abs(conf)
lower = y_pred - abs(conf)

## Scatter plot
plt.clf()
plt.plot(x,y, 'ko', markersize = 1)
plt.title("Scatter plot of birth weight versus mother's age")
plt.xlabel("Mother’s age (y/o)")
plt.ylabel("Baby birth weight (lb)")
plt.show()

plt.clf()
plt.xlim([10, 46])
plt.ylim([6.8, 7.8])
plt.plot(x,y, 'ko', label = "Observation", markersize = 1)
plt.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2, label = r'Regression line (slope = ' + str(round(slope, 3)) + r')')
plt.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4)
 
plt.title("Zoom in on scatter plot of birth weight and mother's age\n\
with regression line and confidence interval")
plt.xlabel("Mother’s age (y/o)")
plt.ylabel("Baby birth weight (lb)")
plt.legend(loc='lower right')
plt.show()

## Percentiles plot
no_na.loc[:, "agepreg_bin"] = " <20"
no_na.loc[(20 <= no_na["agepreg"]) & (no_na["agepreg"] < 25), "agepreg_bin"] = "20-25"
no_na.loc[(25 <= no_na["agepreg"]) & (no_na["agepreg"] < 30), "agepreg_bin"] = "25-30"
no_na.loc[(30 <= no_na["agepreg"]) & (no_na["agepreg"] < 35), "agepreg_bin"] = "30-35"
no_na.loc[(35 <= no_na["agepreg"]) & (no_na["agepreg"] < 40), "agepreg_bin"] = "35-40"
no_na.loc[40 <= no_na["agepreg"], "agepreg_bin"] = ">=40"

no_na.boxplot(column = "totalwgt_lb", by = "agepreg_bin")
#plt.ylim([6, 8.5])
plt.suptitle("")
plt.title("Boxplot on birth weight by mother's age")
plt.xlabel("Mother’s age (y/o)")
plt.ylabel("Baby birth weight (lb)")
plt.show()

## Corr coef
pearson = no_na[["agepreg", "totalwgt_lb"]].corr(method = "pearson")
spearman = no_na[["agepreg", "totalwgt_lb"]].corr(method = "spearman")








