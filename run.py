import numpy as np
import matplotlib.pyplot as plt
from model import *
import seaborn as sns


# Constants definition (Bonds).
T = 60
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
        prices = np.array([MandatoryConvertibleBond('', model=stock_model, alpha=ALPHA, beta=gamma, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price() for c in cs])
        ind_c = np.argmin(abs(prices - P0))
    elif type_bond == "CB":
        prices = np.array([ConvertibleBond('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma).calculate_price() for c in cs]) 
        ind_c = np.argmin(abs(prices - P0))
    else:
        prices = np.array([callableCB('', model=stock_model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma, call_price=F_C).calculate_price() for c in cs])
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
    model = BinomialModel(name="Johnson", delta=1/12, T=60, r=0.05, S_0=175, dividend_dates=[12, 24, 36, 48, 60], dividend_yield=0.0287, U=1.0957, D=0.9127)
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
    strat = CB.strategies[:19,:19]

    import seaborn as sns
    mask = np.zeros_like(strat)
    mask[np.triu_indices_from(mask, k=1)] = True

    ax = sns.heatmap(strat, mask = mask, vmin=-0.5, vmax=3, cmap="YlGnBu", linewidths=.5, cbar=False)
    ax.figure.savefig('heatmap.png')

    for i in range(19):
        for j in range(19): 
            if strat[i,j]==1: 
                print(f'i:{i},j:{j}:: {model.stock_tree[i,j]}')


    # Question 5.3.
    print(cCB.strategies2)
    strat_call = cCB.strategies2[:19,:19]
    mask = np.zeros_like(strat_call)
    mask[np.triu_indices_from(mask, k=1)] = True

    ax = sns.heatmap(strat_call, mask = mask, vmin=-0.5, cmap="YlGnBu", linewidths=.5, cbar=False)
    ax.figure.savefig('heatmap2.png')

    for i in range(19):
        for j in range(19): 
            if strat_call[i,j]==1: 
                print(f'i:{i},j:{j}:: {model.stock_tree[i,j]}')
    
    #print(cCB.question_5_3())
 

    # Question 5.4.
    prices1 = []
    prices2 = []
    prices3 = []
    cs = np.linspace(0.01, 0.2, num=1000)
    for c in cs:
        prices1.append(callableCB('', model=model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA, call_price=F_C).calculate_price())
        prices2.append(MandatoryConvertibleBond('', model=model, alpha=ALPHA, beta=BETA, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price())
        prices3.append(ConvertibleBond('', model=model, face_value=F, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA).calculate_price())

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
    ax[0].plot(cs, prices1, label='callable')
    ax[0].plot(cs, prices2, label='mandatory')
    ax[0].plot(cs, prices3, label='vanilla')
    ax[0].set_xlabel("Coupon rate c")
    ax[0].set_ylabel("Initial price of the CB")
    #ax[0].xaxis.set_tick_params(labelsize=18)
    #ax[0].yaxis.set_tick_params(labelsize=18)
    ax[0].legend()
    ax[0].grid()


    prices1 = []
    prices2 = []
    prices3 = []
    gammas = np.linspace(0.01, 20, num=1000)
    for gamma_ in gammas:
        prices1.append(callableCB('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_, call_price=F_C).calculate_price())
        prices2.append(MandatoryConvertibleBond('', model=model, alpha=ALPHA, beta=gamma_, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]).calculate_price())
        prices3.append(ConvertibleBond('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_).calculate_price())

    ax[1].plot(gammas, prices1, label='callable')
    ax[1].plot(gammas, prices2, label='mandatory')
    ax[1].plot(gammas, prices3, label='vanilla')
    ax[1].set_xlabel(r"Conversion rate $\gamma=\beta$")
    #ax[1].xaxis.set_tick_params(labelsize=18)
    #ax[1].yaxis.set_tick_params(labelsize=18)
    ax[1].legend()
    ax[1].grid() 

    prices = []
    call_price_rates = np.linspace(0.8, 10, num=1000) 
    for call_price_rate in call_price_rates:
        prices.append(callableCB('', model=model, face_value=F, coupon_rate=COUPON_RATE, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=GAMMA, call_price=call_price_rate*F).calculate_price())

    ax[2].plot(call_price_rates, prices, label='callable')
    ax[2].set_xlabel(r"Call price rate $\frac{F_c}{F}$")
    #ax[2].xaxis.set_tick_params(labelsize=18)
    #ax[2].yaxis.set_tick_params(labelsize=18)
    ax[2].grid()
    ax[2].legend()
    plt.show()
    plt.savefig('results/Analysis_5_4.png', dpi=200)


    # Question 5.5.
    P0 = 1000       # We want that each bond is initially worth F.
    cs = []
    cs1 = []
    cs2 = []

    gammas = np.arange(1, 7)
    for gamma_ in gammas:
        cs.append(solve_for_c(model, P0, "cCB", 0, 0.2, 200, gamma_))
        cs1.append(solve_for_c(model, P0, "mCB", 0, 0.2, 200, gamma_))
        cs2.append(solve_for_c(model, P0, "CB", 0, 0.2, 200, gamma_))
   

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    width = 0.2
    ax.bar(gammas-width, cs, width, label='callable')
    ax.bar(gammas, cs1, width,label='mandatory')
    ax.bar(gammas+width, cs2, width, label='vanilla')
    ax.set_xlabel(r"Conversion rate $\gamma$")
    ax.set_ylabel("Coupon rate c")
    #ax.xaxis.set_tick_params(labelsize=18)
    #ax.yaxis.set_tick_params(labelsize=18)
    ax.grid()
    ax.legend()
    plt.show()
    plt.savefig('results/Analysis_5_5.png', dpi=200)
    

    # Question 5.6.
    nb_cs = 50
    nb_call_prices = 20
    cs = np.linspace(0.0, 0.03, num=nb_cs)
    call_price_rates = np.linspace(1, 3, num=nb_call_prices)
    gammas = np.zeros((nb_cs, nb_call_prices))
    for i, c in enumerate(cs):
        for j, call_price_rate in enumerate(call_price_rates):
            gammas[i][j] = solve_for_gamma(model, P0, "cCB", 0, 6, 200, c, call_price_rate)

    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    X, Y = np.meshgrid(cs, call_price_rates)
    cp = plt.contourf(X, Y, gammas.T)
    plt.colorbar(cp)

    ax.set_xlabel(r'Coupon rate $c$')
    ax.set_ylabel(r'Call price rate $\frac{F_c}{F}$')
    plt.show()
