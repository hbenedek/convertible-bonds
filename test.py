import unittest
from model import *
import numpy as np

class TestSum(unittest.TestCase):

    def test_zcb(self):
        model = BinomialModel(name="Lecture4", delta=1, T=3, r=np.log(1.25), S_0=8, dividend_dates=[], dividend_yield=0, U=2, D=1/2)
        model.calculate_risk_neutral_probabilities()
        model.calculate_riskless_tree()
        
        zcb = ZeroCouponBond("zcb",3, 100, model)
        x = zcb.calculate_price()
        self.assertAlmostEqual(x, 51.2)
    

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

if __name__ == '__main__':
    unittest.main()