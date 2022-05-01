### Team

This repository contains the code for the project on Convertible Bonds of the FIN-404 Derivatives course. The team is composed of

   - Santiago Gallego
   - Hilda Abigél Horváth 
   - Benedek Harsányi 

### Dependencies

The project was written with Python 3.8. We used external libraries `matplotlib`, `numpy` and `statsmodels` to generate plots, vectorize some calculations and to perform a linear regression for the model calibration.


### Reproducing Results

In order to reproduce our results for the Analysis (Part 5) one should just run `python3 model.py`. This calls the main function of the file, where the calculations happened. The model calibration happened in the `calibration.py` file. Running this script is possible with `python3 calibration.py`. The dividend yield (delta), riskless interest rate (r), up factor (U) and down factor (D) will be printed out on terminal along with the results of the linear regression.

### Repo Architecture

`model.py` contains our implementation of the BinomialModel along with intermediate calculations of the evolution of the risky asset, the calculation of the EMM probabilities and arbitrage checking. In order to initialize a BM, one should instantiate a BM object with its parameters. This instance is then passed to the corresponding derivative. All derivatives are implemented in the same file as well. A derivative should have an underlying model and a function calculate_price() which returns the price as of today and sets the class attribute price_tree to a np.ndarray matrix, representing the price at different nodes of the BM. We implemented the pricing of simple derivatives (Zero Coupon Bonds, Vanilla Bonds, European Options and American Options ), which we can use as building blocks for pricing more complicated securities such as Mandatory Convertible Bonds, (vanilla) Convertible Bonds and Callable Convertible Bonds.

<pre>  
├─── model.py : Binomial model and Derivative class implementations along with the main executable
├─── test.py : Pricing tests of the different derivative classes
├─── calibration.py : Binomial Model calibration based on J&J historical data
├─── data.txt : Historical data on Johnson & Johnson option prices
├─── README.md 
├─── .gitignore  
└─── description.pdf : contains all specifications and questions of the project
</pre>
