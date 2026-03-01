import numpy as np
import pandas as pd
from scipy.optimize import minimize, brute
from scipy.stats import Normal
from scipy.integrate import quad
import yfinance as yf
import datetime

class StochasticModel:

    def __init__(self, ticker="AAPL", model="black_scholes"):
        self.model = model
        self.ticker = ticker
        self.calib_loop_counter = 0
        self.calib_min_error = 1e9
        self.best_params = None
        print(f"Initialized a {model} model for {ticker}")



    def fetch_market_data(self, expiries=[], min_open_interest=100, atm_threshold=0.1, save_to_class=False, save_to_csv=""):
        if expiries == []: expiries = [self.fetch_expiries()[0]]
        print(f"Fetching market traded options for {self.ticker} for expiry at {expiries}")
        yf_ticker = yf.Ticker(self.ticker)
        last_stock_price = yf_ticker.fast_info["last_price"]
        self.last_stock_price = last_stock_price
        options_df = pd.DataFrame()
        for expiry in expiries:
            opt_chain = yf_ticker.option_chain(expiry)
            calls_df = opt_chain.calls
            calls_df["side"] = "Call"
            calls_df["expiry"] = expiry
            calls_df["T"] = ((datetime.date.fromisoformat(expiry) - datetime.date.today()).days + 1) / 250
            puts_df = opt_chain.puts
            puts_df["expiry"] = expiry
            puts_df["T"] = ((datetime.date.fromisoformat(expiry) - datetime.date.today()).days + 1) / 250
            puts_df["side"] = "Put"
            options_df  = pd.concat([options_df, pd.concat([calls_df, puts_df])])

        liquid_options     = options_df[options_df["openInterest"] > min_open_interest]
        atm_liquid_options = liquid_options[abs(liquid_options["strike"] - last_stock_price) < last_stock_price * atm_threshold]
        atm_liquid_options = atm_liquid_options.reset_index()
        atm_liquid_options = atm_liquid_options[["strike", "lastPrice", "impliedVolatility","side", "expiry", "T"]]

        if save_to_class: self.data = atm_liquid_options
        if save_to_csv != "": atm_liquid_options.to_csv(save_to_csv)

        print(f"Fetched data are filtered for min {min_open_interest} open interest and max {atm_threshold*100}% moneyness threshold")
        return atm_liquid_options



    def fetch_expiries(self):
        print(f"Fetching expiries for traded options of {self.ticker}")
        expiries = yf.Ticker(self.ticker).options
        return expiries



    def load_csv_data(self, file):
        self.data = pd.read_csv(file)



    def fetch_last_stock_price(self):
        print(f"Fetching last underlying price of {self.ticker}")
        yf_ticker = yf.Ticker(self.ticker)
        self.last_stock_price = yf_ticker.fast_info["last_price"]



    def Heston_char_func(self, u, T, r, params):
        # unpacking model params
        kappa_v, theta_v, sigma_v, rho, v0 = params

        c1 = kappa_v * theta_v
        c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v) ** 2 - sigma_v**2 * (-u * 1j - u**2))
        c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (kappa_v - rho * sigma_v * u * 1j - c2)
        H1 = r * u * 1j * T + (c1 / sigma_v**2) * (
            (kappa_v - rho * sigma_v * u * 1j + c2) * T
            - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3)))
        H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v**2
            * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
        
        char_func_value = np.exp(H1 + H2 * v0)
        return char_func_value



    def Lewis_cf_integration(self, u, char_func, S0, K, T, r, params):
        char_func_value = char_func(u - 1j * 0.5, T, r, params)
        cf_integration = (1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real)
        return cf_integration



    def Merton_char_func(self, u, T, params):
        lamb, mu, delta = params
        omega = -lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
        char_func_value = np.exp(
            (1j * u * omega + lamb * (np.exp(1j * u * mu - u**2 * delta**2 * 0.5) - 1)) * T)
        return char_func_value



    def Bates_char_func(self, u, T, r, params):
        kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = params
        params_Heston = kappa_v, theta_v, sigma_v, rho, v0
        params_Merton = lamb, mu, delta

        H_cf = self.Heston_char_func(u, T, r, params_Heston)
        M_cf = self.Merton_char_func(u, T, params_Merton)
        return H_cf * M_cf



    def price_option(self, option_side="Call", S0=None, K=None, T=None, r=None, params=None):
        assert S0 is not None and K is not None and T is not None and r is not None and params is not None, \
                "Parameters 'S0', 'K', 'r' and 'T' are required for pricing"

        if self.model == "black_scholes":
            sigma = params
            d1 = (np.log(S0/K) + ((r + (sigma**2) / 2) * T)) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            norm_cdf = Normal(mu=0, sigma=1).cdf
            if option_side == "Call":
                call_price = S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
                return max(call_price,0)
            else:
                # for simplicity, I use put-call parity for pricing put options
                put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
                return max(put_price, 0)

        elif self.model == "heston":
            integration_value = quad(lambda u: self.Lewis_cf_integration(u, self.Heston_char_func, 
                                                                        S0, K, T, r, params),0, np.inf, limit=250)[0]
            call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * integration_value)

            if option_side == "Call":
                return call_value
            else:
                # for simplicity, I use put-call parity for pricing put options
                return max(0, call_value + K * np.exp(-r * T) - S0)

        elif self.model == "bates":
            integration_value = quad(lambda u: self.Lewis_cf_integration(u, self.Bates_char_func,
                                                                        S0, K, T, r, params), 0, np.inf, limit=250)[0]
            call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * integration_value)

            if option_side == "Call":
                return call_value
            else:
                # for simplicity, I use put-call parity for pricing put options
                return max(0, call_value + K * np.exp(-r * T) - S0)



    def batch_price_option(self, data, r=None, params=None, return_column=""):
        assert data is not None and r is not None and params is not None, \
            "Parameters 'data', 'r' and 'params' are required for pricing"

        prices = []

        for _, option in data.iterrows():
            price = self.price_option(option["side"], self.last_stock_price, option["strike"], option["T"], r, params)
            prices.append(price)

        if return_column != "":
            data[return_column] = prices
        else:
            return prices



    def error_func(self, market_data, r, params, error_type ="mse", print_step=50, print_report=False):
        model_predictions = pd.Series(self.batch_price_option(market_data, r, params))
        if error_type == "mae":
            step_error = np.mean(abs(market_data["lastPrice"] - model_predictions) / market_data["lastPrice"])
        elif error_type == "mse":
            step_error = np.mean(((market_data["lastPrice"] - model_predictions) / market_data["lastPrice"]) ** 2)

        if step_error < self.calib_min_error:
            self.calib_min_error = step_error
            self.best_params = params

        if self.calib_loop_counter % print_step == 0 and print_report:
            print(f"- Step {self.calib_loop_counter}:  best params: {self.best_params}  |  Min error: {self.calib_min_error.round(3)}")

        self.calib_loop_counter += 1
        return step_error
    


    def quick_calibration(self, market_data, r, input_ranges, max_calls, error_type="mse", print_step=50, print_report=False):
        self.calib_loop_counter = 0
        self.calib_min_error = 1e9
        self.fetch_last_stock_price()
        steps = 1
        while (steps + 1) ** len(input_ranges) <= max_calls:
            steps += 1

        print("=" * 70)
        print(f"Quick calibration over {steps} steps, total {steps ** len(input_ranges)} calls...")
        quick_results = brute(lambda x: self.error_func(market_data, r, x, error_type, print_step, print_report), input_ranges, Ns=steps, finish=None)
        print("=" * 70)
        print(" Lowest error reached: ", self.calib_min_error)
        print(" Optimal parameters:   ", self.best_params)
        return quick_results



    def calibrate(self, market_data, r, x0, error_type="mse", print_step=50, print_report=False, bounds=[]):
        
        self.calib_loop_counter = 0
        self.calib_min_error = 1e9
        self.fetch_last_stock_price()

        if bounds == []:
            if self.model == "black_scholes":
                bounds = [(0.001, 2)]
            elif self.model == "heston":
                bounds = [(0.05, 5), (0.01, 2), (0.01, 2), (-1, 1), (0.01, 0.5)]
            elif self.model == "bates":
                bounds = [(0.05, 5), (0.01, 2), (0.01, 2), (-1, 0.5), (0.01, 0.5),
                        (0.02,4), (-0.3,0.3), (0.05, 0.5)]

        print("=" * 70)
        print(f"Calibrating the function...")
        calib_results = minimize(lambda x: self.error_func(market_data, r, x, error_type, print_step, print_report), x0=x0, bounds=bounds, method="L-BFGS-B")
        print("=" * 70)
        print(" Lowest error reached: ", calib_results.fun)
        print(" Optimal parameters:   ", calib_results.x)
        return calib_results



