import numpy as np


class Binomial:
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
        self.risk_neutral_down = self.risk_neutral_up - 1

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


class ZeroCouponBond(Binomial):
    def __init__(self, name: str, maturity: int, face_value: float):
        super().__init__()
        self.name = name
        self.maturity = maturity
        self.face_value = face_value

    def calculate_price(self) -> float:
        return self.face_value * np.exp(self.r * self.delta * self.maturity)


class PlainCouponBond(Binomial):
    def __init__(self, name: str, maturity: int, face_value: float, coupon_rate: float):
        super().__init__()
        self.name = name
        self.maturity = maturity
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon = self.coupon_rate * self.face_value

    def calculate_price(self) -> float:
        # decompose the bond into zero coupon bonds and principle payments
        coupons = sum([ZeroCouponBond(f"{self.name}-ZCB-{i}", i ,self.coupon).calculate_price() for i in range(1,self.maturity + 1)])
        principal = ZeroCouponBond(f"{self.name}-principal-{self.maturity}", self.maturity , self.face_value).calculate_price()
        return coupons + principal 


class EuropeanCall(Binomial):
    def __init__(self, name, K): 
        super().__init__()
        self.name = name
        self.K = K

    def calculate_price(self) -> float:
        pass
        
class EuropeanPut(Binomial):
    pass
    def calculate_price(self) -> float:
        pass

class MandatoryConvertibleBond(Binomial):
    pass
    def calculate_price(self) -> float:
        pass


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    model = Binomial("test", 1, 5, 2, 1, 0, 0, 2, 1/2)
    model.calculate_risk_neutral_probabilities()
    model.check_arbitrage()
    model.calculate_stock_tree()
    print(model.stock_tree)
    model.calculate_riskless_tree()
    print(model.riskless_tree)