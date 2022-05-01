import numpy as np
import matplotlib.pyplot as plt
from model import *


# Constants definition (Bonds).
ALPHA = 1000
BETA = 4
GAMMA = 4       # Conversion rate.
COUPON_RATE = 0.02
COUPON_DATES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
F = 1000
F_C = 1.05*F    # Call price.


def solve_for_c(stock_model: BinomialModel, P0: float, type_bond: str, c_min: float, c_max: float, nb_iter: float, gamma: float):
    cs = np.linspace(c_min, c_max, nb_iter)

    ind_c = None
    if type_bond == "mCB": 
        prices = np.array([MandatoryConvertibleBond('', model=stock_model, alpha=ALPHA, beta=gamma, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price() for gamma_ in gammas])
        ind_c = np.argmin(abs(prices - P0))
    elif type_bond == "CB":
        prices = np.array([ConvertibleBond('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma).calculate_price() for gamma_ in gammas]) 
        ind_c = np.argmin(abs(prices - P0))
    else:
        prices = np.array([callableCB('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma, call_price=F_C).calculate_price() for gamma_ in gammas])
        ind_c = np.argmin(abs(prices - P0))

    return cs[ind_c]


def solve_for_gamma(stock_model: BinomialModel, P0: float, type_bond: str, gamma_min: float, gamma_max: float, nb_iter: float, c: float, call_price_rate: float):
    gammas = np.linspace(gamma_min, gamma_max, nb_iter)

    ind_gamma = None
    if type_bond == "mCB": 
        prices = np.array([MandatoryConvertibleBond('', model=stock_model, alpha=ALPHA, beta=gamma_, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price() for gamma_ in gammas])
        ind_gamma = np.argmin(abs(prices - P0))
    elif type_bond == "CB":
        prices = np.array([ConvertibleBond('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_).calculate_price() for gamma_ in gammas]) 
        ind_gamma = np.argmin(abs(prices - P0))
    else:
        prices = np.array([callableCB('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_, call_price=call_price_rate*F).calculate_price() for gamma_ in gammas])
        ind_gamma = np.argmin(abs(prices - P0))

    return gammas[ind_gamma]


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    model = BinomialModel(name="Lecture4", delta=1/12, T=60, r=0.05, S_0=175, dividend_dates=[12, 24, 36, 48, 60], dividend_yield=0.0287, U=1.0957, D=0.9127)
    model.calculate_risk_neutral_probabilities()
    model.check_arbitrage()
    model.calculate_stock_tree()
    model.calculate_riskless_tree()
    
    #Question 5.1.
    mCB = MandatoryConvertibleBond('', model=model, alpha=ALPHA, beta=BETA, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    print('Mandatory CB:', mCB.calculate_price())

    CB = ConvertibleBond('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA) 
    print('CB:', CB.calculate_price())

    cCB = callableCB('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA, call_price=F_C)
    B0 = cCB.calculate_price()
    print('callable CB:', B0)

       # Question 5.2.
    #TODO:


    # Question 5.3.
    #print(cCB.question_5_3())
 

    # Question 5.4.
    prices = []
    cs = np.linspace(0.01, 0.99, num=1000)
    for c in cs:
        prices.append(callableCB('', model=model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA, call_price=F_C).calculate_price())
        # prices.append(MandatoryConvertibleBond('', model=model, alpha=ALPHA, beta=BETA, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price())
        # prices.append(ConvertibleBond('', model=model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA).calculate_price())
    
    plt.plot(cs, prices)
    plt.xlabel("Coupon rate c")
    plt.ylabel("Initial price of the cCB")
    plt.grid()
    plt.show()
    
    prices = []
    gammas = np.linspace(0.01, 30, num=1000)
    for gamma_ in gammas:
        # prices.append(MandatoryConvertibleBond('', model=model, alpha=ALPHA, beta=gamma_, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price())
        # prices.append(ConvertibleBond('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_).calculate_price())
        prices.append(callableCB('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_, call_price=F_C).calculate_price())
    
    plt.plot(gammas, prices)
    plt.xlabel(r"Conversion rate $\gamma$")
    plt.ylabel("Initial price of the cCB")
    plt.grid()
    plt.show() 

    prices = []
    call_price_rates = np.linspace(0.8, 10, num=1000) 
    for call_price_rate in call_price_rates:
        prices.append(callableCB('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA, call_price=call_price_rate*F).calculate_price())

    plt.plot(call_price_rates, prices)
    plt.xlabel(r"Call price rate $\frac{F_c}{F}$")
    plt.ylabel("Initial price of the cCB")
    plt.grid()
    plt.show() 


    # Question 5.5.
    P0 = 1000       # We want that each bond is initially worth F.
    cs = []
    gammas = np.arange(1, 7)
    for gamma_ in gammas:
        cs.append(solve_for_c(model, P0, "cCB", 0, 0.2, 1000, gamma_))

    plt.bar(gammas, cs)
    plt.xlabel(r"Conversion rate $\gamma$")
    plt.ylabel("Coupon rate c")
    plt.grid()
    plt.show()  
    


    # Question 5.6.
    nb_cs = 10
    nb_call_prices = 10
    cs = np.linspace(0.01, 0.99, num=nb_cs)
    call_price_rates = np.linspace(0.8, 10, num=nb_call_prices)
    gammas = np.zeros((nb_cs, nb_call_prices))
    for i, c in enumerate(cs):
        #TODO: add code for other bonds outside the second for loop.

        for j, call_price_rate in enumerate(call_price_rates):
            gammas[i][j] = solve_for_gamma(model, P0, "cCB", 0.01, 30, 100, c, call_price_rate)

    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
 
    # Creating plot
    x = np.outer(cs, np.ones(nb_cs))
    y = np.outer(call_price_rates, np.ones(nb_call_prices)).T # transpose
   
    ax.plot_surface(x, y, gammas)
    ax.set_xlabel("Coupon rate c")
    ax.set_ylabel(r"Call price rate $\frac{F_c}{F}$")
    ax.set_zlabel(r"Conversion rate $\gamma$")
    plt.show()