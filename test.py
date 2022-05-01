import unittest
from model import *
import numpy as np

class Test(unittest.TestCase):

    def test_zcb(self):
        model = BinomialModel(name="Lecture4", delta=1, T=3, r=np.log(1.25), S_0=8, dividend_dates=[], dividend_yield=0, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_riskless_tree()
        
        zcb = ZeroCouponBond("zcb",3, 100, model)
        x = zcb.calculate_price()
        self.assertAlmostEqual(x, 51.2)
    
    def test_coupon_bond(self):
        model = BinomialModel(name="Lecture4", delta=1, T=2, r=np.log(1.25), S_0=4, dividend_dates=[1], dividend_yield=0.25, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_riskless_tree()

        bond = PlainCouponBond('bond', 2, 100, 0.03, [1,2], model)
        x = bond.calculate_price()
        self.assertAlmostEqual(x, 68.32)

    def test_european_call(self):
        model = BinomialModel(name="Lecture3", delta=1, T=3, r=np.log(1.25), S_0=8, dividend_dates=[], dividend_yield=0, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_stock_tree()
        model.calculate_riskless_tree()

        call = EuropeanOption('test_call', K=40, model=model, type_ = 'call')
        x = call.calculate_price()
        self.assertAlmostEqual(x, 1.536)


    def test_american_put(self):
        model = BinomialModel(name="Lecture4", delta=1, T=2, r=np.log(1.25), S_0=4, dividend_dates=[], dividend_yield=0, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_stock_tree()
        model.calculate_riskless_tree()

        put = AmericanOption('test_put', model=model, K=5, payoff=np.empty(0) , type_='put')
        x = put.calculate_price()
        self.assertEqual(x, 1.36)


    def test_mCB(self):
        model = BinomialModel(name="Lecture6", delta=1, T=1, r=np.log(1.25), S_0=4, dividend_dates=[], dividend_yield=0.25, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_stock_tree()
        model.calculate_riskless_tree()
        
        mCB = MandatoryConvertibleBond('',alpha=100,beta=5,model=model,face_value=20, coupon_rate=0.02, coupon_dates=[1]) 
        x = mCB.calculate_price()
        self.assertAlmostEqual(x, 24.32)


    def test_CB(self):
        model = BinomialModel(name="Lecture6", delta=1, T=1, r=np.log(1.25), S_0=4, dividend_dates=[], dividend_yield=0.25, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_stock_tree()
        model.calculate_riskless_tree()

        CB = ConvertibleBond('', model=model, face_value=20, coupon_rate=0.02, coupon_dates=[1], gamma=10) 
        x = CB.calculate_price()
        self.assertAlmostEqual(x, 40.320) 

    def test_convertibleCB(self):
        model = BinomialModel(name="Lecture6", delta=1, T=2, r=np.log(1.25), S_0=4, dividend_dates=[1], dividend_yield=0.25, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_stock_tree()
        model.calculate_riskless_tree()

        cCB = callableCB('test_cCB', model=model, face_value=20, coupon_rate=0.02, coupon_dates=[1,2], gamma=10, call_price=21)
        B0 = cCB.calculate_price()
        self.assertEqual(B0, 40)

if __name__ == '__main__':
    unittest.main()