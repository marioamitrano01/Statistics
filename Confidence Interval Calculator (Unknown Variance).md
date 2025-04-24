# Advanced Confidence Interval Calculator (Unknown Variance)

This project contains a Python script that calculates **confidence intervals** for data when the population variance is **unknown** (e.g., small samples, or whenever you rely on the **t-distribution** rather than the **z-distribution**). Specifically, it calculates:

1. **Confidence Interval for the Mean**  
   - Uses the **t-distribution** to get the lower and upper bounds for the sample mean.

2. **Confidence Interval for the Variance**  
   - Uses the **chi-square distribution** to estimate how the true variance might vary around your sample variance.

3. **Confidence Interval for the Standard Deviation**  
   - Derived from the variance interval by taking the square roots of the lower and upper bounds.

4. **Shapiro-Wilk Normality Test**  
   - Checks if your data are likely to come from a **normal distribution** (a key assumption for using the t and chi-square formulas). Shows a **W-statistic** and a **p-value**.

5. **Histogram and Q–Q Plot**  
   - **Histogram**: Visual display of your data’s distribution, including vertical lines for the mean’s confidence interval bounds.  
   - **Q–Q Plot**: Helps you assess visually if the data follow a normal distribution (ideally, points lie close to a straight line).

### How It Works
- **Collects user input**:  
  - A space-separated list of numbers (the sample).  
  - A confidence level (like 0.95 or 0.99).

- **Computes**:  
  - Sample mean, variance, and standard deviation.  
  - Confidence intervals around mean, variance, and standard deviation.  
  - Runs a **Shapiro-Wilk** test for normality.

- **Displays**:  
  - All results in a text format in your console.  
  - Optional **plots** using `matplotlib` (a histogram for the distribution and a Q–Q plot for normality).

### Why This Matters
- **Unknown Variance**: In many real-world scenarios, the population variance is not known, so you rely on sample statistics to estimate it.  
- **More Accurate for Smaller Samples**: Traditional “z-based” confidence intervals assume you know the variance or have a large sample. The **t-distribution** is more appropriate when your sample is small and the variance is estimated from that sample.  
- **Normality Check**: Since these intervals assume data are normally distributed (or nearly so), it’s good practice to verify it with the **Shapiro-Wilk** test and a **Q–Q plot**.
