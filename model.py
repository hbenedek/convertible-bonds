from abc import abstractclassmethod
import numpy as np


class BinomialModel:
    def __init__(self, name: str, delta: float, T: int, r: float, S_0: float, dividend_dates: float, dividend_yield: float, U: float, D: float):
        self.name = name
        self.delta = delta
        self.T = T
        self.r = r
        self.S_0 = S_0
        self.dividend_rates = dividend_dates
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
            for j in range(0,i):
                prices[i][j] = prices[i - 1][j] * self.U
            prices[i][i] = prices[i - 1][i - 1] * self.D
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
        self.model = None

    def calculate_price(self) -> float:
        return self.face_value * np.exp(self.model.r * self.model.delta * self.maturity)


class PlainCouponBond(Derivative):
    def __init__(self, name: str, maturity: int, face_value: float, coupon_rate: float, model: BinomialModel):
        super().__init__(name, model)
        self.maturity = maturity
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon = self.coupon_rate * self.face_value

    def calculate_price(self) -> float:
        # decompose the bond into zero coupon bonds and principle payments
        coupons = sum([ZeroCouponBond(f"{self.name}-ZCB-{i}", i ,self.coupon, self.model).calculate_price() for i in range(1, self.maturity + 1)])
        principal = ZeroCouponBond(f"{self.name}-principal-{self.maturity}", self.maturity , self.face_value, self.model).calculate_price()
        return coupons + principal 


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

        
class EuropeanPut(Derivative):
    pass
    def calculate_price(self) -> float:
        pass

class MandatoryConvertibleBond:
    pass
    def calculate_price(self) -> float:
        pass


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    model = BinomialModel(name="Lecture4", delta=1, T=3, r=np.log(1.25), S_0=8, dividend_dates=0, dividend_yield=0, U=2, D=1/2)
    model.calculate_risk_neutral_probabilities()
    model.check_arbitrage()
    model.calculate_stock_tree()
    model.calculate_riskless_tree()
    call = EuropeanOption('test_call', K=40, model=model, type_ = 'call')
    x = call.calculate_price()
    print(call.price_tree)
    print(x)