from abc import abstractclassmethod
import numpy as np
import matplotlib.pyplot as plt


class BinomialModel:
    def __init__(self, name: str, delta: float, T: int, r: float, S_0: float, dividend_dates: list, dividend_yield: float, U: float, D: float):
        self.name = name
        self.delta = delta
        self.T = T
        self.r = r
        self.S_0 = S_0
        self.dividend_dates = dividend_dates
        self.dividend_yield = dividend_yield
        self.U = U
        self.D = D
        self.risk_neutral_up = None
        self.risk_neutral_down = None
        self.stock_tree = None
        self.riskless_tree = None

    def calculate_risk_neutral_probabilities(self):
        self.risk_neutral_up = (np.exp(self.r * self.delta) - self.D) / (self.U - self.D)
        self.risk_neutral_down = 1 - self.risk_neutral_up

    def check_arbitrage(self):
        if self.risk_neutral_up < 1 and 0 < self.risk_neutral_up: 
            print(f"{self.name} is arbitrage free with risk neutral probabilities: {self.risk_neutral_up} and {self.risk_neutral_down}") 
        else:
            print(f"WARNING: {self.name} is not arbitrage free: q={self.risk_neutral_up}")

    def calculate_stock_tree(self):
        prices = np.zeros((self.T + 1, self.T + 1))
        prices[0][0] = self.S_0
        for i in range(1, self.T + 1):
            delta_i = self.dividend_yield if (i in self.dividend_dates) else 0          # Dividend yield at date i.
            
            for j in range(0,i):
                Sc_i = prices[i - 1][j] * self.U                                        # Cum-value.
                prices[i][j] = Sc_i*(1 - delta_i)                                       # Ex-dividend value.
            Sc_i = prices[i - 1][i - 1] * self.D
            prices[i][i] = Sc_i*(1 - delta_i)
        self.stock_tree = prices

    def calculate_riskless_tree(self):
        prices = np.zeros((self.T + 1, self.T + 1))
        current = 1
        prices[0][0] = current
        for i in range(1, self.T + 1):
            current *= np.exp(self.r * self.delta)
            for j in range(0,i + 1):
                prices[i][j] = current
        self.riskless_tree = prices


class Derivative:
    def __init__(self, name: str, model: BinomialModel):
        self.name = name
        self.model = model

    @abstractclassmethod
    def calculate_price():
        pass


class ZeroCouponBond(Derivative):
    def __init__(self, name: str, maturity: int, face_value: float, model: BinomialModel):
        super().__init__(name, model)
        self.maturity = maturity
        self.face_value = face_value
        self.price_tree = None

    def calculate_price(self) -> float:
        gamma = np.exp(- self.model.r * self.model.delta)
        mask = np.ones(self.maturity + 1)
        prices = np.zeros((self.maturity + 1, self.maturity + 1))
        prices[self.maturity] = self.face_value
        for i in range(self.maturity-1, -1, -1):
            mask[i+1:] = 0
            prices[i] = prices[i + 1] * gamma * mask
        self.price_tree = prices
        return self.price_tree[0][0]


class PlainCouponBond(Derivative):
    def __init__(self, name: str, maturity: int, face_value: float, coupon_rate: float, coupon_dates: list, model: BinomialModel):
        super().__init__(name, model)
        self.maturity = maturity
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_dates = coupon_dates
        self.coupon = self.coupon_rate * self.face_value
        self.price_tree = None

    def calculate_price(self) -> float:
        prices = np.zeros((self.maturity + 1, self.maturity + 1))

        # decompose the bond into zero coupon bonds and principle payments
        coupons = [ZeroCouponBond(f"{self.name}-ZCB-{i}", i ,self.coupon, self.model) for i in self.coupon_dates]
        
        for coupon in coupons:
            coupon.calculate_price()
            tmp = coupon.price_tree
            tmp[-1,:] = 0               # Mask the prices at maturity (in order to get the ex-coupon prices). Comment this line in order to get cum-coupon prices.
            prices += np.pad(coupon.price_tree, [(0, prices.shape[0] - coupon.price_tree.shape[0]), (0, prices.shape[1] - coupon.price_tree.shape[1])], mode='constant')

        principal = ZeroCouponBond(f"{self.name}-principal-{self.maturity}", self.maturity , self.face_value, self.model)
        principal.calculate_price()
        prices += principal.price_tree

        self.price_tree = prices
        return self.price_tree[0][0]


class EuropeanOption(Derivative):
    def __init__(self, name: str, K: float, model: BinomialModel, type_: str): 
        super().__init__(name, model)
        self.K = K
        self.price_tree = None
        self.type = type_
        if self.type == 'call':
            self.payoff = lambda x: max(0, x - self.K)
        elif self.type == 'put':
            self.payoff = lambda x: max(0, self.K - x)

    def calculate_price(self) -> float:
        prices = np.zeros((self.model.T + 1, self.model.T + 1))
        gamma = np.exp(-self.model.r * self.model.delta)
        for i in range(self.model.T + 1):
            prices[self.model.T][i] = self.payoff(self.model.stock_tree[self.model.T][i])
        for j in range(self.model.T - 1, -1, -1):
            for i in range(0, j + 1):
                prices[j][i] = gamma * (self.model.risk_neutral_up * prices[j + 1][i + 1] + self.model.risk_neutral_down * prices[j + 1][i])
        self.price_tree = prices
        return self.price_tree[0][0]

class MandatoryConvertibleBond(Derivative):
    def __init__(self, name: str, alpha: float, beta: float, model: BinomialModel, face_value: float, coupon_rate: float, coupon_dates):
        super().__init__(name, model)
        self.alpha = alpha
        self.beta = beta
        self.face_value = face_value
        self.coupon_rate  = coupon_rate
        self.coupon_dates = coupon_dates
        self.call_price_tree = None
        self.put_price_tree = None
        self.bond_price_tree = None
        self.price_tree = None

    def calculate_price(self) -> float:
        call = EuropeanOption("call", self.face_value/self.beta, self.model, "call")
        put = EuropeanOption("put", self.face_value/self.alpha, self.model, "put")
        bond = PlainCouponBond("bond", self.model.T, self.face_value, self.coupon_rate, self.coupon_dates, self.model) 

        call.calculate_price()
        put.calculate_price()
        bond.calculate_price()

        self.call_price_tree = call.price_tree
        self.put_price_tree = put.price_tree
        self.bond_price_tree = bond.price_tree

        self.price_tree = self.beta * self.call_price_tree - self.alpha * self.put_price_tree + self.bond_price_tree
        return self.price_tree[0][0]
        

class AmericanOption(Derivative):
    def __init__(self, name: str, model: BinomialModel, K: float = 0, payoff: np.ndarray = np.empty(0), type_: str = ""):
        super().__init__(name, model)
        self.price_tree = None
        self.type = type_
        self.K = K
        if self.type == 'call':
            call = lambda x: max(0, x - self.K)
            self.payoff = np.vectorize(call)(self.model.stock_tree)
        elif self.type == 'put':
            put = lambda x: max(0, self.K - x)
            self.payoff = np.vectorize(put)(self.model.stock_tree)
        else:
            self.payoff = payoff


    def calculate_price(self) -> float:
        prices = np.zeros((self.model.T + 1, self.model.T + 1))
        gamma = np.exp(-self.model.r * self.model.delta)
        prices[self.model.T] = self.payoff[self.model.T]
        for j in range(self.model.T - 1, -1, -1):
            for i in range(0, j + 1):
                continuation = gamma * (self.model.risk_neutral_up * prices[j + 1][i + 1] + self.model.risk_neutral_down * prices[j + 1][i])
                prices[j][i] = max(continuation, self.payoff[j][i])

        self.price_tree = prices
        return self.price_tree[0][0]

class ConvertibleBond(Derivative): 
    def __init__(self, name: str,  model: BinomialModel, face_value: float, coupon_rate: float, coupon_dates, gamma: float):
        super().__init__(name, model)
        self.face_value = face_value
        self.coupon_rate  = coupon_rate
        self.coupon_dates = coupon_dates
        self.gamma = gamma
        self.american_price_tree = None
        self.bond_price_tree = None
        self.price_tree = None

    def calculate_price(self) -> float:
        # calculate payoff
        payoff = np.zeros(self.model.T) 
        payoff[self.model.T] = np.maximum(self.face_value, self.gamma*self.model.stock_tree[self.model.T])
        for i in range(self.model.T): 
            vtB = PlainCouponBond("coup", self.model.T-i, self.face_value, self.coupon_rate, self.model)
            vtB_price = vtB.calculate_price()
            payoff[i] = self.gamma*self.model.stock_tree[self.model.T] - vtB_price

        # replicating portfolio
        american = AmericanOption("am", model, payoff=payoff)
        bond = PlainCouponBond("bond", self.model.T, self.face_value, self.coupon_rate, self.model)

        american.calculate_price()
        bond.calculate_price()

        self.american_price_tree = american.price_tree
        self.bond_price_tree = bond.price_tree

        # calculate price
        self.price_tree = self.american_price_tree + self.bond_price_tree
        return self.price_tree[0][0]


class action:
    def __init__(self, date: int, actor: str, stock_value: float):
        self.date = date
        self.actor = actor
        self.stock_value = stock_value
    
    def describe(self) -> str:
        description = "\nAt date t = {0}, if the share price S_t = {1}\n".format(self.date, self.stock_value)
        if (self.actor == "both"):
            description += "   The issuer should call the bond and the bondholder should exercise the conversion option.\n"
        elif (self.actor == "bondholder"):
            description += "   The {} should exercise the conversion option.\n".format(self.actor)   
        else:
            description += "   The {} should exercise the call option.\n".format(self.actor)
        return description


class callableCB(Derivative):
    def __init__(self, name: str,  model: BinomialModel, face_value: float, coupon_rate: float, coupon_dates: list, gamma: float, call_price: float):
        super().__init__(name, model)
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_dates = coupon_dates
        self.gamma = gamma                                                                              # Conversion rate.
        self.call_price = call_price
        self.price_tree = None
        self.strategies = []

    def calculate_price(self) -> float:
        self.price_tree = np.zeros((self.model.T + 1, self.model.T + 1))

        # Calculate payoff at maturity
        for i in range(self.model.T+1):
            S_T = self.model.stock_tree[self.model.T][i]

            if (self.face_value > max(self.gamma*S_T, self.call_price)):
                price = max(self.gamma*S_T, self.call_price) 
                self.price_tree[self.model.T][i] = price                                                # Update price tree.

                # Update strategies list.
                if price == self.gamma*S_T:     self.strategies.append(action(self.model.T, "both", S_T))
                else:                           self.strategies.append(action(self.model.T, "issuer", S_T))
            else:
                price = max(self.gamma*S_T, self.face_value)
                self.price_tree[self.model.T][i] = price

                # Update strategies list.
                if price == self.gamma*S_T:     self.strategies.append(action(self.model.T, "bondholder", S_T))
        
        for i in range(self.model.T - 1, -1, -1):
            coupon_i = self.coupon_rate*self.face_value if ((i+1) in self.coupon_dates) else 0          # Coupon payment at date j.
            for j in range(i+1):
                S_i = self.model.stock_tree[i][j]                                                       # i-th value of stock at date j.
                # Continuation value.
                C_i = np.exp(-self.model.r * self.model.delta)*(self.model.risk_neutral_up*self.price_tree[i+1][j] + self.model.risk_neutral_down*self.price_tree[i+1][j+1] + coupon_i)

                if ((i in self.coupon_dates) and (C_i > max(self.gamma*S_i, self.call_price))):
                    price = max(self.gamma*S_i, self.call_price)
                    self.price_tree[i][j] = price

                    if price == self.gamma*S_i: self.strategies.append(action(i, "both", S_i))
                    else:                       self.strategies.append(action(i, "issuer", S_i))

                else:
                    price = max(self.gamma*S_i, C_i)
                    self.price_tree[i][j] = price
                    
                    if price == self.gamma*S_i:     self.strategies.append(action(i, "bondholder", S_i))

        return self.price_tree[0][0]
        
    def describe_strategies(self) -> str:
        descriptions = '\n\nN.B. In principle, only the first occurrence (of the exercise of the call and/or conversion option) should be taken into account. '
        descriptions += 'However, we list "subsequent" strategies, in case the first one is not followed (e.g. case where the bondholder decided to continue, even if it was not the optimal decision).\n'
        for i in range(len(self.strategies)-1, -1, -1):
            descriptions += self.strategies[i].describe()

        return descriptions

    def question_5_3(self) -> str:
        descriptions = ""
        for i in range(len(self.strategies)-1, -1, -1):
            if self.strategies[i].date in [6, 12, 18]:
                descriptions += self.strategies[i].describe()
        
        return descriptions





if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    model = BinomialModel(name="Lecture4", delta=1/12, T=60, r=0.05, S_0=175, dividend_dates=[12, 24, 36, 48, 60], dividend_yield=0.0287, U=1.0957, D=0.9127)
    model.calculate_risk_neutral_probabilities()
    model.check_arbitrage()
    model.calculate_stock_tree()
    model.calculate_riskless_tree()
    
    """
    zcb = ZeroCouponBond("zcb",3, 100, model)
    zcb.calculate_price()
    print(zcb.price_tree)
    """

    #bond = PlainCouponBond('test', 2, 100, 0.03, model)
    #bond.calculate_price()
    #cb = PlainCouponBond("cb", 3, 1, 0.05, [1,2,3], model)
    # print(cb.calculate_price())
    # print(cb.price_tree)
    

    cCB = callableCB('test_cCB', model=model, face_value=1000, coupon_rate=0.02, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=4, call_price=1050)
    B0 = cCB.calculate_price()
    print(B0)
    #print(cCB.question_5_3())
 
    # Question 5.4.
    cCB_prices = []
    for c in np.linspace(0.01, 0.99, num=100):
        cCB_prices.append(callableCB('', model=model, face_value=1000, coupon_rate=c, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=4, call_price=1050).calculate_price())
    
    plt.plot(np.linspace(0.01, 0.99, num=100), cCB_prices)
    plt.xlabel("coupon rate c")
    plt.ylabel("Initial price of the cCB")
    plt.grid()
    plt.show()
    
    cCB_prices = []
    for gamma_ in np.linspace(0.01, 30, num=100):
        cCB_prices.append(callableCB('', model=model, face_value=1000, coupon_rate=0.02, coupon_dates=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], gamma=gamma_, call_price=1050).calculate_price())
    
    plt.plot(np.linspace(0.01, 30, num=100), cCB_prices)
    plt.xlabel("conversion rate gamma")
    plt.ylabel("Initial price of the cCB")
    plt.grid()
    plt.show() 

