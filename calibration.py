import pandas as pd
import numpy as np
from model import *
import statsmodels.formula.api as smf

# data 
data = pd.read_csv('data.txt', header=0,  sep="\t")
print(data.columns)

# assumptions: r is constant, dividends each year with delta
S_0 = 175

# using put call parity
# call - put - S_0

data["val"]  = data["call"].values - data["put"].values - np.ones(len(data["call"])) * S_0

# linear regression 
model = smf.ols(formula='val ~ 1 + strike', data=data)
results = model.fit()
print("Result of the linear regression")
print(results.summary())

disc_factor = -results.params[1]
PVdividend = -results.params[0]

# calculate r 
r = -1 * np.log(disc_factor)
print("Value of r:")
print(r)

# caluculate delta
x = PVdividend / S_0
delta = x / (1+x)
print("Value of delta:")
print(delta)

# calibrate model
# 60 one-month period, q = 1/2

D = np.exp(r/12) - np.sqrt(np.exp(r/6) - 1)  
U = 1/D
print("Value of D:")
print(D)
print("Value of U")
print(U)

BinomialModel("Johnson", delta=1/12, T=60, r=r, S_0=S_0, dividend_dates=[12,24,36,48,60], dividend_yield=delta, U=U, D=D)
