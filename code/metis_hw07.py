#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:13:53 2017

@author: katharina
"""

import pandas as pd
import numpy as np
import scipy as sp
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








