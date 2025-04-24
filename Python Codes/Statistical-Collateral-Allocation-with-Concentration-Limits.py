import time
import dataclasses
from typing import List, Dict, Tuple, Optional
import functools 
import random
import pulp
import plotly.graph_objs as go
import plotly.offline as pyo



def print_colored_paragraph(text: str, color_code: str) -> None:
    """
    Prints a paragraph of text in a given ANSI color, then resets the color.
    
    """
    reset_code = "\033[0m"
    print(f"{color_code}{text}{reset_code}\n")

COLOR_BLUE    = "\033[94m"
COLOR_YELLOW  = "\033[93m"
COLOR_GREEN   = "\033[92m"
COLOR_CYAN    = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED     = "\033[91m"




def timing_decorator(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        highlight = COLOR_RED if elapsed > 1 else COLOR_MAGENTA
        print(f"{highlight}Function '{func.__name__}' completed in {elapsed:.6f} seconds.\033[0m")
        return result
    return wrapper

def log_call_decorator(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] +
                            [f"{k}={v!r}" for k,v in kwargs.items()])
        print_colored_paragraph(
            f"[LOG] Calling {func.__name__}({arg_str})",
            COLOR_CYAN
        )
        result = func(*args, **kwargs)
        print_colored_paragraph(
            f"[LOG] {func.__name__} returned {result!r}",
            COLOR_CYAN
        )
        return result
    return wrapper



HAIRCUT_MAPPING: Dict[str, float] = {
    "cash equities": 0.30,
    "exotics derivatives and structured derivatives": 0.65,
    "investment grade bond": 0.10,
    "emerging markets bond": 0.25,
    "g10 currency spot": 0.00
}

def get_haircut(ctype: str) -> float:
    return HAIRCUT_MAPPING.get(ctype.lower().strip(), 0.0)




@dataclasses.dataclass
class Collateral:
    
    name: str                 # e.g. "Cash Equities"
    total_amount: float       # maximum number of units available
    cost: float               # cost per unit of collateral
    ctype: str                # textual category, e.g. "Cash Equities"
    lot_size: float = 1.0
    currency: str = "USD"
    fx_rate_to_usd: float = 1.0

    def coverage_per_unit(self) -> float:
        
        haircut = get_haircut(self.ctype)
        return (1.0 - haircut) * self.fx_rate_to_usd

@dataclasses.dataclass
class CSA:
    
    name: str
    required_margin_usd: float
    eligibility: Dict[str, bool] = dataclasses.field(default_factory=dict)



class CollateralAllocationModel:
    
    def __init__(
        self,
        collateral_list: List[Collateral],
        csa_list: List[CSA],
        max_fraction_map: Optional[Dict[str, float]] = None
    ):
        """
        :param collateral_list: list of available collateral objects
        :param csa_list: list of CSAs with margin requirements
        :param max_fraction_map: dictionary mapping ctype.lower() -> max fraction
           Example: {"g10 currency spot":0.1, ...}
           If None, no concentration limit is applied for that
        """
        self.collateral_list = collateral_list
        self.csa_list = csa_list
        self.max_fraction_map = max_fraction_map or {}
        self.model: Optional[pulp.LpProblem] = None
        self.allocation_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}

    @timing_decorator
    @log_call_decorator
    def build_model(self) -> None:
        self.model = pulp.LpProblem("Collateral_Allocation", pulp.LpMinimize)

        for i, col in enumerate(self.collateral_list):
            for j, csa in enumerate(self.csa_list):
                var_name = f"x_{col.name}_{csa.name}"
                self.allocation_vars[(i, j)] = pulp.LpVariable(
                    var_name, lowBound=0, cat=pulp.LpContinuous
                )

        cost_expr = []
        for (i, j), var in self.allocation_vars.items():
            col = self.collateral_list[i]
            cost_expr.append(var * col.cost)
        self.model += pulp.lpSum(cost_expr), "Minimize_Total_Cost"

        for i, col in enumerate(self.collateral_list):
            self.model += (
                pulp.lpSum(self.allocation_vars[(i, j)] for j in range(len(self.csa_list)))
                <= col.total_amount,
                f"Avail_{col.name}"
            )

        for j, csa in enumerate(self.csa_list):
            coverage_terms = []
            for i, col in enumerate(self.collateral_list):
                c_lower = col.ctype.lower()
                var_ij = self.allocation_vars[(i, j)]
                coverage_expr = var_ij * col.coverage_per_unit()

                if not csa.eligibility.get(c_lower, False):
                    self.model += (
                        var_ij <= 0,
                        f"Inelig_{col.name}_{csa.name}"
                    )
                else:
                    coverage_terms.append(coverage_expr)

                    if c_lower in self.max_fraction_map:
                        max_frac = self.max_fraction_map[c_lower]
                        self.model += (
                            coverage_expr <= max_frac * csa.required_margin_usd,
                            f"ConcLimit_{col.name}_{csa.name}"
                        )

            self.model += (
                pulp.lpSum(coverage_terms) >= csa.required_margin_usd,
                f"ReqMargin_{csa.name}"
            )

    @timing_decorator
    @log_call_decorator
    def solve(self) -> Optional[Dict[Tuple[str, str], float]]:
        if self.model is None:
            print_colored_paragraph("Error: build_model() must be called before solve()", COLOR_RED)
            return None

        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        status_str = pulp.LpStatus[self.model.status]

        if status_str == "Optimal":
            print_colored_paragraph("[Model] An optimal solution was found!", COLOR_GREEN)
            result: Dict[Tuple[str, str], float] = {}
            for (i, j), var in self.allocation_vars.items():
                val = var.varValue
                if val and val > 1e-9:
                    csa_name = self.csa_list[j].name
                    col_name = self.collateral_list[i].name
                    result[(col_name, csa_name)] = val
            return result
        else:
            print_colored_paragraph(f"[Model] No optimal solution. Status = {status_str}", COLOR_RED)
            return None





@timing_decorator
def run_allocation_with_concentration_limit() -> None:
    

    print_colored_paragraph("===== CSA WITH CONCENTRATION LIMIT =====", COLOR_BLUE)

    while True:
        try:
            n_csas = int(input("How many CSAs do you want to create? ").strip())
            if n_csas < 1:
                print_colored_paragraph("Please enter an integer >= 1.", COLOR_RED)
                continue
            break
        except ValueError:
            print_colored_paragraph("Invalid integer. Try again.", COLOR_RED)

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

    csa_list: List[CSA] = []
    for idx in range(1, n_csas + 1):
        while True:
            inp = input(f"Margin requirement (USD) for CSA{idx}: ")
            try:
                margin_req = float(inp)
                if margin_req < 0:
                    print_colored_paragraph("Margin cannot be negative!", COLOR_RED)
                    continue
                break
            except ValueError:
                print_colored_paragraph("Invalid numeric value. Try again.", COLOR_RED)

        csa_obj = CSA(
            name=f"CSA{idx}",
            required_margin_usd=margin_req,
            eligibility=global_eligibility
        )
        csa_list.append(csa_obj)

    print_colored_paragraph("=== Summary of Created CSAs ===", COLOR_CYAN)
    for csa in csa_list:
        print(f"  {csa.name}: margin = {csa.required_margin_usd} USD (all classes are eligible)")

    total_requirement = sum(c.required_margin_usd for c in csa_list)
    buffer_factor = 2.0  # can adjust to something bigger if needed
    collateral_list: List[Collateral] = []

    for ac in asset_classes:
        col_name = ac.replace(" ", "")
       
        coverage_capacity_needed = total_requirement / (1.0 - get_haircut(ac.lower()))
        total_amount = buffer_factor * coverage_capacity_needed

        coll = Collateral(
            name=col_name,
            total_amount=total_amount,
            cost=0.01,  # same cost for demonstration
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
        print_colored_paragraph("Error: The model is infeasible or no optimal solution found!", COLOR_RED)
        return

    print_colored_paragraph("=== ALLOCATION PLAN (non-zero postings) ===", COLOR_GREEN)
    for (col_name, csa_name), amount in allocation_result.items():
        print(f"  {col_name} -> {csa_name}: {amount:.2f}")

    print_colored_paragraph("Generating stacked bar chart in your browser...", COLOR_MAGENTA)

    csa_names = [c.name for c in csa_list]
    data_map = {c.name: {col.name: 0.0 for col in collateral_list} for c in csa_list}
    for (col_n, csa_n), val in allocation_result.items():
        data_map[csa_n][col_n] = val

    bars = []
    for col in collateral_list:
        y_vals = [data_map[csa.name][col.name] for csa in csa_list]
        bars.append(go.Bar(x=csa_names, y=y_vals, name=col.name))

    layout = go.Layout(
        title="Collateral Allocation with Concentration Limits",
        barmode='stack',
        xaxis=dict(title='CSA'),
        yaxis=dict(title='Collateral Units Allocated')
    )
    fig = go.Figure(data=bars, layout=layout)
    pyo.plot(fig)

    print("Done! Check your browser for the Mario's interactive plot.")



if __name__ == "__main__":
    run_allocation_with_concentration_limit()
