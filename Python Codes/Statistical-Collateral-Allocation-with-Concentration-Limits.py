import time
import dataclasses
from typing import List, Dict, Tuple, Optional
import functools 
import pulp
import plotly.graph_objs as go
import plotly.offline as pyo
from dataclasses import field


COLOR_BLUE    = "\033[94m"
COLOR_YELLOW  = "\033[93m"
COLOR_GREEN   = "\033[92m"
COLOR_CYAN    = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED     = "\033[91m"
RESET_CODE    = "\033[0m"


def print_colored(text: str, color_code: str) -> None:
    print(f"{color_code}{text}{RESET_CODE}")


HAIRCUT_MAPPING = {
    "cash equities": 0.30,
    "exotics derivatives and structured derivatives": 0.65,
    "investment grade bond": 0.10,
    "emerging markets bond": 0.25,
    "g10 currency spot": 0.00
}


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        highlight = COLOR_RED if elapsed > 1 else COLOR_MAGENTA
        print(f"{highlight}Function '{func.__name__}' completed in {elapsed:.6f} seconds.{RESET_CODE}")
        return result
    return wrapper


def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] +
                            [f"{k}={v!r}" for k, v in kwargs.items()])
        print_colored(f"[LOG] Calling {func.__name__}({arg_str})", COLOR_CYAN)
        result = func(*args, **kwargs)
        print_colored(f"[LOG] {func.__name__} returned {result!r}", COLOR_CYAN)
        return result
    return wrapper


@dataclasses.dataclass
class Collateral:
    name: str                 
    total_amount: float      
    cost: float               
    ctype: str                
    lot_size: float = 1.0
    currency: str = "USD"
    fx_rate_to_usd: float = 1.0
    
    _coverage: float = field(init=False)
    
    def __post_init__(self):
        haircut = HAIRCUT_MAPPING.get(self.ctype.lower().strip(), 0.0)
        self._coverage = (1.0 - haircut) * self.fx_rate_to_usd
    
    def coverage_per_unit(self) -> float:
        return self._coverage


@dataclasses.dataclass
class CSA:
    name: str
    required_margin_usd: float
    eligibility: Dict[str, bool] = field(default_factory=dict)


class CollateralAllocationModel:
    def __init__(
        self,
        collateral_list: List[Collateral],
        csa_list: List[CSA],
        max_fraction_map: Optional[Dict[str, float]] = None
    ):
        
        self.collateral_list = collateral_list
        self.csa_list = csa_list
        self.max_fraction_map = max_fraction_map or {}
        self.model = None
        self.allocation_vars = {}
        self.solver = pulp.PULP_CBC_CMD(msg=0)  

    @timing_decorator
    def build_model(self) -> None:
        self.model = pulp.LpProblem("Collateral_Allocation", pulp.LpMinimize)
        num_collaterals = len(self.collateral_list)
        num_csas = len(self.csa_list)
        
        eligibility_matrix = {}
        for i, col in enumerate(self.collateral_list):
            c_lower = col.ctype.lower()
            for j, csa in enumerate(self.csa_list):
                eligibility_matrix[(i, j)] = csa.eligibility.get(c_lower, False)
        
        for i, col in enumerate(self.collateral_list):
            for j, csa in enumerate(self.csa_list):
                if eligibility_matrix[(i, j)]:
                    var_name = f"x_{i}_{j}" 
                    self.allocation_vars[(i, j)] = pulp.LpVariable(
                        var_name, lowBound=0, cat=pulp.LpContinuous
                    )
        
        self.model += pulp.lpSum(
            self.allocation_vars.get((i, j), 0) * self.collateral_list[i].cost
            for i in range(num_collaterals) for j in range(num_csas)
            if (i, j) in self.allocation_vars
        ), "Minimize_Total_Cost"
        
        for i, col in enumerate(self.collateral_list):
            self.model += (
                pulp.lpSum(
                    self.allocation_vars.get((i, j), 0)
                    for j in range(num_csas)
                    if (i, j) in self.allocation_vars
                ) <= col.total_amount,
                f"Avail_{i}"
            )
        
        for j, csa in enumerate(self.csa_list):
            coverage_terms = []
            
            for i, col in enumerate(self.collateral_list):
                if (i, j) not in self.allocation_vars:
                    continue
                
                var_ij = self.allocation_vars[(i, j)]
                c_lower = col.ctype.lower()
                coverage_expr = var_ij * col.coverage_per_unit()
                coverage_terms.append(coverage_expr)
                
                if c_lower in self.max_fraction_map:
                    max_frac = self.max_fraction_map[c_lower]
                    self.model += (
                        coverage_expr <= max_frac * csa.required_margin_usd,
                        f"ConcLimit_{i}_{j}"
                    )
            
            self.model += (
                pulp.lpSum(coverage_terms) >= csa.required_margin_usd,
                f"ReqMargin_{j}"
            )

    @timing_decorator
    def solve(self) -> Optional[Dict[Tuple[str, str], float]]:
        if self.model is None:
            print_colored("Error: build_model() must be called before solve()", COLOR_RED)
            return None

        self.model.solve(self.solver)
        status_str = pulp.LpStatus[self.model.status]

        if status_str == "Optimal":
            print_colored("[Model] An optimal solution was found!", COLOR_GREEN)
            result = {}
            
            for (i, j), var in self.allocation_vars.items():
                val = var.value()
                if val and val > 1e-9:  
                    csa_name = self.csa_list[j].name
                    col_name = self.collateral_list[i].name
                    result[(col_name, csa_name)] = val
            
            return result
        else:
            print_colored(f"[Model] No optimal solution. Status = {status_str}", COLOR_RED)
            return None


def get_user_integer(prompt: str, min_value: int = 1) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value < min_value:
                print_colored(f"Please enter an integer >= {min_value}.", COLOR_RED)
                continue
            return value
        except ValueError:
            print_colored("Invalid integer. Try again.", COLOR_RED)


def get_user_float(prompt: str, min_value: float = 0) -> float:
    while True:
        try:
            value = float(input(prompt).strip())
            if value < min_value:
                print_colored(f"Value cannot be less than {min_value}!", COLOR_RED)
                continue
            return value
        except ValueError:
            print_colored("Invalid numeric value. Try again.", COLOR_RED)


@timing_decorator
def run_allocation_with_concentration_limit() -> None:
    print_colored("CSA WITH CONCENTRATION LIMIT", COLOR_BLUE)

    n_csas = get_user_integer("How many CSAs do you want to create? ")

    asset_classes = [
        "Investment Grade Bond",
        "Cash Equities",
        "G10 Currency Spot",
        "Emerging Markets Bond",
        "Exotics Derivatives and Structured Derivatives"
    ]
    
    max_fraction_map = {
        "investment grade bond": 0.50,
        "cash equities": 0.10,
        "g10 currency spot": 0.10,
        "emerging markets bond": 0.20,
        "exotics derivatives and structured derivatives": 0.10
    }

    global_eligibility = {ac.lower(): True for ac in asset_classes}

    csa_list = []
    for idx in range(1, n_csas + 1):
        margin_req = get_user_float(f"Margin requirement (USD) for CSA{idx}: ")
        csa_obj = CSA(
            name=f"CSA{idx}",
            required_margin_usd=margin_req,
            eligibility=global_eligibility
        )
        csa_list.append(csa_obj)

    print_colored("=== Summary of Created CSAs ===", COLOR_CYAN)
    for csa in csa_list:
        print(f"  {csa.name}: margin = {csa.required_margin_usd} USD (all classes are eligible)")

    total_requirement = sum(c.required_margin_usd for c in csa_list)
    buffer_factor = 2.0
    collateral_list = []

    for ac in asset_classes:
        col_name = ac.replace(" ", "")
        haircut = HAIRCUT_MAPPING.get(ac.lower(), 0.0)
        coverage_capacity_needed = total_requirement / (1.0 - haircut)
        total_amount = buffer_factor * coverage_capacity_needed

        coll = Collateral(
            name=col_name,
            total_amount=total_amount,
            cost=0.01,  
            ctype=ac
        )
        collateral_list.append(coll)

    model = CollateralAllocationModel(
        collateral_list=collateral_list,
        csa_list=csa_list,
        max_fraction_map=max_fraction_map
    )
    
    model.build_model()
    allocation_result = model.solve()
    
    if allocation_result is None:
        print_colored("Error: The model is infeasible or no optimal solution found!", COLOR_RED)
        return

    print_colored(" ALLOCATION PLAN (non-zero postings) ", COLOR_GREEN)
    for (col_name, csa_name), amount in allocation_result.items():
        print(f"  {col_name} -> {csa_name}: {amount:.2f}")

    print_colored("Generating stacked bar chart in your browser...", COLOR_MAGENTA)

    csa_names = [c.name for c in csa_list]
    collateral_names = [c.name for c in collateral_list]
    
    data_map = {csa_n: {col_n: 0.0 for col_n in collateral_names} for csa_n in csa_names}
    
    for (col_n, csa_n), val in allocation_result.items():
        data_map[csa_n][col_n] = val

    bars = []
    for col_name in collateral_names:
        y_vals = [data_map[csa_n][col_name] for csa_n in csa_names]
        bars.append(go.Bar(x=csa_names, y=y_vals, name=col_name))

    layout = go.Layout(
        title="Collateral Allocation with Concentration Limits",
        barmode='stack',
        xaxis=dict(title='CSA'),
        yaxis=dict(title='Collateral Units Allocated')
    )
    
    fig = go.Figure(data=bars, layout=layout)
    pyo.plot(fig)

    print("Done! Check your browser for the interactive plot.")


if __name__ == "__main__":
    run_allocation_with_concentration_limit()
