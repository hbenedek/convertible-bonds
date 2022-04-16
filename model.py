from calendar import c
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


class ZeroCouponBond(Binomial):
    def __init__(self, name: str, maturity: int, face_value: float):
        self.name = name
        self.maturity = maturity
        self.face_value = face_value

    def calculate_price(self) -> float:
        return self.face_value * np.exp(self.r * self.delta * self.maturity)


class PlainCouponBond(Binomial):
    def __init__(self, name: str, maturity: int, face_value: float, coupon_rate: float):
        self.name = name
        self.maturity = maturity
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon = self.coupon_rate * self.face_value

    def calculate_price(self) -> float:
       coupons = sum([ZeroCouponBond(f"{self.name}-ZCB-{i}", i ,self.coupon).calculate_price() for i in range(1,self.maturity + 1)])
       principal = ZeroCouponBond(f"{self.name}-principal-{self.maturity}", self.maturity , self.face_value).calculate_price()
       return coupons + principal 


class EuropeanCall(Binomial):
    pass

class EuropeanPut(Binomial):
    pass

class MandatoryConvertibleBond(Binomial):
    pass