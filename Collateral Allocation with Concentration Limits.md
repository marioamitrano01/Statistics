# Collateral Allocation with Concentration Limits

**Objective:**  
Minimize total collateral posting cost while satisfying margin requirements for multiple CSAs (Credit Support Annexes) and adhering to:  
- **Haircut** (coverage) constraints  
- **Collateral availability**  
- **Concentration** limits  

---

## 1. Overview

1. **Multiple CSAs** each require a certain USD margin.  
2. **Collateral Pool** with different asset types, costs, and FX rates.  
3. **Constraints:**
   - **Haircuts** reduce effective coverage.  
   - **Availability** limits total units posted.  
   - **Concentration** caps how much of each CSA’s margin can come from a single asset type.  

---

## 2. Main Components

1. **Decorators**  
   - `timing_decorator`: Measures execution time.  
   - `log_call_decorator`: Logs function calls & return values.

2. **Haircuts**  
   - `HAIRCUT_MAPPING`: Maps asset types to haircut percentages.  
   - `get_haircut()`: Retrieves haircut for a given collateral type.

3. **Data Classes**  
   - `Collateral`: Name, available amount, cost, type, FX rate → provides coverage in USD (after haircut).  
   - `CSA`: Name, required margin, eligibility (which assets allowed).

4. **CollateralAllocationModel**  
   - **Decision Variables** 
   - **Objective**  
   - **Constraints**:
     1. **Availability**
     2. **Eligibility** 
     3. **Coverage**
     4. **Concentration**

5. **Main**  
   - `run_allocation_with_concentration_limit()`:  
     1. Requests number of CSAs and their margin amounts.  
     2. Creates a large pool of collateral.  
     3. Defines concentration limits.  
     4. Builds & solves the model.  
     5. Displays results in text and a stacked bar chart (Plotly).


## 3. Usage

1. **Run Script**  
2. **Follow Prompts**  
   - Enter number of CSAs.  
   - Enter margin requirement for each.  
3. **View Allocation**  
   - Text output in the terminal.  
   - Stacked bar chart opens in your browser.
