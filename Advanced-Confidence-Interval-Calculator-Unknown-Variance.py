import time
import logging
from functools import wraps
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import statistics
import scipy.stats as stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def time_method(method):
    """
    Decorator for timing the execution of a particular method.
    Logs the elapsed time for performance analysis.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = method(self, *args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"Method '{method.__name__}' executed in {elapsed:.6f} seconds.")
        return result
    return wrapper


@dataclass
class ConfidenceIntervalUnknownVariance:
    
    sample: List[float]
    power: int
    confidence_level: float
    n: int = field(init=False)
    df: int = field(init=False)

    _mean: float = field(init=False, default=0.0)
    _s2: float = field(init=False, default=0.0)
    _s2_c: float = field(init=False, default=0.0)
    _s_c: float = field(init=False, default=0.0)
    _t_crit: float = field(init=False, default=0.0)
    _chi_sq_lower: float = field(init=False, default=0.0)
    _chi_sq_upper: float = field(init=False, default=0.0)

    _mean_lower: float = field(init=False, default=0.0)
    _mean_upper: float = field(init=False, default=0.0)
    _var_lower: float = field(init=False, default=0.0)
    _var_upper: float = field(init=False, default=0.0)

    _shapiro_stat: float = field(init=False, default=0.0)
    _shapiro_p_value: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        
        self.n = len(self.sample)
        self.df = self.n - 1

        self._mean = statistics.mean(self.sample)
        self._s2 = self._calculate_s2()
        # S^2_c = scaled sample variance = (N / (N-1)) * S^2
        self._s2_c = (self.n / self.df) * self._s2
        self._s_c = np.sqrt(self._s2_c)

        self._shapiro_stat, self._shapiro_p_value = stats.shapiro(self.sample)
        logging.info(
            f"Shapiro-Wilk Test: W-stat = {self._shapiro_stat:.4f}, "
            f"p-value = {self._shapiro_p_value:.4g}"
        )

        half_alpha = (1.0 - self.confidence_level) / 2.0
        self._t_crit = stats.t.ppf(1.0 - half_alpha, df=self.df)
        self._chi_sq_lower = stats.chi2.ppf(half_alpha, df=self.df)
        self._chi_sq_upper = stats.chi2.ppf(1.0 - half_alpha, df=self.df)

        self.calculate_confidence_limits()

    def _calculate_s2(self) -> float:
        
        avg_power = np.mean([x ** self.power for x in self.sample])
        return avg_power - (self._mean ** self.power)

    @time_method
    def calculate_confidence_limits(self) -> None:
       
        margin_of_error = (self._s_c / np.sqrt(self.n)) * self._t_crit
        self._mean_lower = self._mean - margin_of_error
        self._mean_upper = self._mean + margin_of_error

        self._var_lower = (self.df * self._s2_c) / self._chi_sq_upper
        self._var_upper = (self.df * self._s2_c) / self._chi_sq_lower

    @property
    def mean_confidence_interval(self) -> Tuple[float, float]:
        
        return (self._mean_lower, self._mean_upper)

    @property
    def variance_confidence_interval(self) -> Tuple[float, float]:
        
        return (self._var_lower, self._var_upper)

    @property
    def std_dev_confidence_interval(self) -> Tuple[float, float]:
        
        return (np.sqrt(self._var_lower), np.sqrt(self._var_upper))

    @property
    def mean(self) -> float:
        
        return self._mean

    @property
    def s2(self) -> float:
        return self._s2

    @property
    def s2_c(self) -> float:
        return self._s2_c

    @property
    def s_c(self) -> float:
        return self._s_c

    @property
    def shapiro_wilk(self) -> Tuple[float, float]:
        
        return self._shapiro_stat, self._shapiro_p_value

    @time_method
    def display_results(self) -> None:
        
        print("= Advanced Confidence Interval Results =")
        print(f"Sample Size (n):               {self.n}")
        print(f"Sample Mean:                   {self._mean:.4f}")
        print(f"Unscaled S^2:                  {self._s2:.4f}")
        print(f"Scaled Variance (S^2_c):       {self._s2_c:.4f}")
        print(f"Estimated Std. Dev. (S_c):     {self._s_c:.4f}")
        print(f"Mean Interval (t-based):       [{self._mean_lower:.4f}, {self._mean_upper:.4f}]")
        print(f"Variance Interval (χ²-based):  [{self._var_lower:.4f}, {self._var_upper:.4f}]")
        lower_std, upper_std = self.std_dev_confidence_interval
        print(f"Std. Dev. Interval:            [{lower_std:.4f}, {upper_std:.4f}]")
        print("\n--- Normality Test (Shapiro-Wilk) ---")
        print(f" W-statistic: {self._shapiro_stat:.4f}")
        print(f" p-value:     {self._shapiro_p_value:.4g}")
        print("------")

    @time_method
    def plot_sample_and_intervals(self) -> None:
        """
        Create a histogram of the sample with vertical lines indicating 
        the mean and its confidence interval bounds.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(self.sample, bins='auto', color='skyblue', alpha=0.7, edgecolor='black')
        
        ax.axvline(self._mean, color='red', linestyle='--',
                   label=f'Mean = {self._mean:.2f}')
        ax.axvline(self._mean_lower, color='green', linestyle='--',
                   label=f'Mean Lower = {self._mean_lower:.2f}')
        ax.axvline(self._mean_upper, color='green', linestyle='--',
                   label=f'Mean Upper = {self._mean_upper:.2f}')
        
        ax.set_title("Sample Distribution with Mean Confidence Interval")
        ax.set_xlabel("Sample Values")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.show()

    @time_method
    def plot_qq(self) -> None:
        
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(self.sample, dist="norm", plot=ax)
        ax.set_title("Q–Q Plot for Normality Assessment")
        plt.show()


def main() -> None:
    
    logging.info("Welcome to the Advanced Confidence Interval Calculator for Unknown Variance!")
    while True:
        print("\nOptions:")
        print("1. Calculate Confidence Intervals")
        print("2. Exit")

        choice = input("Select an option: ").strip()
        if choice == '1':
            try:
                raw_data = input("Enter your sample (space-separated, e.g. 1.4 5.7 3.41): ")
                sample_list = [float(x) for x in raw_data.split()]
                confidence_lvl = float(input("Enter desired confidence level (0.0 < level < 1.0): "))
                
                if not sample_list:
                    raise ValueError("The sample cannot be empty.")
                if not (0.0 < confidence_lvl < 1.0):
                    raise ValueError("Confidence level must be between 0.0 and 1.0.")
                
                ci_calc = ConfidenceIntervalUnknownVariance(
                    sample=sample_list,
                    power=2, 
                    confidence_level=confidence_lvl
                )
                
                ci_calc.display_results()

                ci_calc.plot_sample_and_intervals()

                user_qq = input("Do you want to see a Q–Q plot for normality check? (y/n): ").strip().lower()
                if user_qq.startswith('y'):
                    ci_calc.plot_qq()

            except ValueError as e:
                logging.error(f"Invalid input: {e}")
            break

        elif choice == '2':
            logging.info("Exiting. Goodbye!")
            break

        else:
            print("Please select either '1' or '2'.")


if __name__ == "__main__":
    main()
