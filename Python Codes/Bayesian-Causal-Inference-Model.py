import numpy as np
import pymc as pm
import arviz as az
import plotly.graph_objects as go
import plotly.express as px
import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper


class BayesianCausalInference:
    def __init__(self, N, p_trt, true_alpha_control, true_alpha_treatment,
                 true_beta, true_gamma, sigma, seed):
        
    
        self.N = N
        self.p_trt = p_trt
        self.true_alpha_control = true_alpha_control
        self.true_alpha_treatment = true_alpha_treatment
        self.true_beta = true_beta
        self.true_gamma = true_gamma
        self.sigma = sigma
        self.seed = seed

        self.trt = None       
        self.conf = None      
        self.y = None         

        self.trace = None    
        self.model = None     

        np.random.seed(self.seed)

    @timer
    def simulate_data(self):
        
        self.trt = np.random.binomial(1, self.p_trt, size=self.N)
        self.conf = np.random.normal(0, 1, size=self.N)
        alpha = np.where(self.trt == 1, self.true_alpha_treatment, self.true_alpha_control)
        mu = alpha + self.true_beta * self.trt + self.true_gamma * self.conf
        self.y = np.random.normal(mu, self.sigma)
        print("Data simulation complete.")

    @timer
    def build_model(self):
        
        with pm.Model() as self.model:
            alpha_control = pm.Normal("alpha_control", mu=0, sigma=10)
            alpha_treatment = pm.Normal("alpha_treatment", mu=0, sigma=10)
            
            beta = pm.Normal("beta", mu=0, sigma=10)
            
            gamma = pm.Normal("gamma", mu=0, sigma=10)
            
            sigma = pm.HalfCauchy("sigma", beta=5)
            
            alpha = pm.math.switch(pm.math.eq(self.trt, 1), alpha_treatment, alpha_control)
            
            mu_val = alpha + beta * self.trt + gamma * self.conf
            
            y_obs = pm.Normal("y_obs", mu=mu_val, sigma=sigma, observed=self.y)
        print("Model built successfully.")

    @timer
    def run_inference(self, draws, tune, chains, target_accept):
        
        if self.model is None:
            raise ValueError("You must first build the model by calling build_model().")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=self.seed
            )
        print("Inference completed.")
        print(az.summary(self.trace, var_names=["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]))

    def plot_posteriors(self):
        
        if self.trace is None:
            raise ValueError("Run inference first with run_inference().")
        
        param_names = ["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]
        fig = go.Figure()
        for param in param_names:
            samples = self.trace.posterior[param].values.flatten()
            hist_data = np.histogram(samples, bins=50, density=True)
            bin_centers = 0.5 * (hist_data[1][1:] + hist_data[1][:-1])
            fig.add_trace(go.Scatter(x=bin_centers, y=hist_data[0],
                                     mode='lines',
                                     name=param))
        
        fig.update_layout(title="Posterior Distributions of Parameters",
                          xaxis_title="Parameter Value",
                          yaxis_title="Density",
                          template="plotly_white")
        fig.show()

    def plot_trace(self):
        
        if self.trace is None:
            raise ValueError("Run inference first with run_inference().")
        
        param_names = ["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]
        fig = go.Figure()
        for param in param_names:
            chains = self.trace.posterior[param].chain.values
            for ch in chains:
                samples = self.trace.posterior[param].sel(chain=ch).values.flatten()
                fig.add_trace(go.Scatter(y=samples,
                                         mode='lines',
                                         name=f"{param} - chain {ch}"))
        fig.update_layout(title="MCMC Trace Plots",
                          yaxis_title="Parameter Value",
                          template="plotly_white")
        fig.show()

    def posterior_predictive_check(self, num_pp_samples):
        
        if self.model is None or self.trace is None:
            raise ValueError("Ensure that you have built the model and run inference before calling this method.")
        
        trace_subset = self.trace.copy()
        trace_subset.posterior = trace_subset.posterior.sel(draw=slice(0, num_pp_samples))
        
        with self.model:
            ppc = pm.sample_posterior_predictive(
                trace_subset,
                var_names=["y_obs"],
                random_seed=self.seed,
                return_inferencedata=False  
            )
        
        y_ppc = ppc["y_obs"].flatten()
        
       
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_ppc,
            nbinsx=50,
            opacity=0.5,
            name="Posterior Predictions",
            marker_color='red',
            marker_line_color='black',
            marker_line_width=1.0
        ))
        fig.add_trace(go.Histogram(
            x=self.y,
            nbinsx=50,
            opacity=0.5,
            name="Observed Data",
            marker_color='blue',
            marker_line_color='black',
            marker_line_width=1.0
        ))
        fig.update_layout(barmode='overlay',
                          title="Posterior Predictive Check",
                          xaxis_title="y value",
                          yaxis_title="Count",
                          template="plotly_white")
        fig.show()


if __name__ == "__main__":
    print("Welcome to the Mario's Bayesian Causal Inference Model.")
    print("Please follow the instructions to input the required parameters.\n")
    
    try:
        N = int(input("Enter total number of observations (e.g., 1000): "))
        p_trt = float(input("Enter probability of treatment assignment (0 to 1, e.g., 0.5): "))
        true_alpha_control = float(input("Enter true intercept for control group (e.g., 2.0): "))
        true_alpha_treatment = float(input("Enter true intercept for treatment group (e.g., 2.0): "))
        true_beta = float(input("Enter true causal effect of the treatment (e.g., 3.0): "))
        true_gamma = float(input("Enter true coefficient for the confounding variable (e.g., 1.5): "))
        sigma = float(input("Enter the standard deviation of the outcome errors (e.g., 1.0): "))
        seed = int(input("Enter a random seed (e.g., 42): "))
        
        draws = int(input("Enter number of MCMC draws (post-tuning, e.g., 2000): "))
        tune = int(input("Enter number of tuning steps (e.g., 1000): "))
        chains = int(input("Enter number of MCMC chains (e.g., 4): "))
        target_accept = float(input("Enter target acceptance rate for NUTS sampler (e.g., 0.9): "))
        num_pp_samples = int(input("Enter number of posterior predictive samples per chain (e.g., 200): "))
    except Exception as e:
        print("Invalid input. Please restart and enter valid numerical values.")
        raise e

    model_instance = BayesianCausalInference(
        N=N,
        p_trt=p_trt,
        true_alpha_control=true_alpha_control,
        true_alpha_treatment=true_alpha_treatment,
        true_beta=true_beta,
        true_gamma=true_gamma,
        sigma=sigma,
        seed=seed
    )
    
    model_instance.simulate_data()
    
    model_instance.build_model()
    
    model_instance.run_inference(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    
    model_instance.plot_posteriors()
    
    model_instance.plot_trace()
    
    model_instance.posterior_predictive_check(num_pp_samples=num_pp_samples)
    
    print("Bayesian causal inference complete. Explore the interactive plots for detailed diagnostics.")
