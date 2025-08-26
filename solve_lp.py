"""
Linear Programming for Resource Optimization
Two ready-to-run models in one script:
  1) Factory Production Planning (LP)
  2) Workforce Shift Scheduling (MILP)

Dependencies: pulp (pip install pulp)
How to run: python lp_resource_optimization.py
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pulp as pl

# ============================
# 1) FACTORY PRODUCTION PLANNING (LP)
# ============================


@dataclass
class Product:
    profit: float
    demand_max: float  # use a large number if unlimited


@dataclass
class Resource:
    capacity: float

# consumption[(resource, product)] = units of resource consumed per unit product


def solve_production_planning(products: Dict[str, Product],
                              resources: Dict[str, Resource],
                              consumption: Dict[Tuple[str, str], float],
                              verbose: bool = True):
    model = pl.LpProblem("ProductionPlanning", pl.LpMaximize)

    # Decision variables: quantity of each product
    x = {p: pl.LpVariable(f"x_{p}", lowBound=0) for p in products}

    # Objective: maximize total profit
    model += pl.lpSum(products[p].profit * x[p]
                      for p in products), "TotalProfit"

    # Resource capacity constraints
    cons_resource = {}
    for r in resources:
        cons = pl.lpSum(consumption.get((r, p), 0.0) *
                        x[p] for p in products) <= resources[r].capacity
        cname = f"cap_{r}"
        model += cons, cname
        cons_resource[r] = model.constraints[cname]

    # Demand limits
    cons_demand = {}
    for p in products:
        cname = f"dem_{p}"
        model += x[p] <= products[p].demand_max, cname
        cons_demand[p] = model.constraints[cname]

    # Solve
    status = model.solve(pl.PULP_CBC_CMD(msg=False))

    # Collect results
    result = {
        "status": pl.LpStatus[status],
        "objective": pl.value(model.objective),
        "x": {p: x[p].value() for p in products},
        "resource_slack": {},
        "resource_dual": {},
        "binding_resources": [],
    }

    # Slack and duals (shadow prices) when available
    for r, c in cons_resource.items():
        slack = c.slack
        dual = getattr(c, "pi", None)  # CBC exposes .pi
        result["resource_slack"][r] = slack
        result["resource_dual"][r] = dual
        if abs(slack) < 1e-7:
            result["binding_resources"].append(r)

    if verbose:
        print("\n=== Production Planning Result ===")
        print("Status:", result["status"])
        print("Optimal Profit:", round(result["objective"], 4))
        print("Quantities:")
        for p, v in result["x"].items():
            print(f"  {p}: {v:.4f}")
        print("\nResource Utilization:")
        for r in resources:
            used = sum(consumption.get((r, p), 0.0) *
                       result["x"][p] for p in products)
            cap = resources[r].capacity
            slack = result["resource_slack"][r]
            dual = result["resource_dual"][r]
            print(
                f"  {r}: used={used:.4f} / cap={cap}, slack={slack:.4f}, shadow_price={None if dual is None else round(dual, 4)}")
        if result["binding_resources"]:
            print("Binding resources:", ", ".join(result["binding_resources"]))
        else:
            print("No binding resources.")

    return result

# ============================
# 2) WORKFORCE SHIFT SCHEDULING (MILP)
# ============================
# Binary assignment x[person, shift] = 1 if person works that shift
# Objective: minimize total wage cost while meeting required coverage per shift


def solve_staffing(people: List[str],
                   shifts: List[str],
                   coverage_required: Dict[str, int],
                   cost_per_person_per_shift: Dict[str, float],
                   # 1 if available, else 0
                   availability: Dict[Tuple[str, str], int],
                   max_shifts_per_person: int = 5,
                   forbid_consecutive: List[Tuple[str, str]] = None,
                   verbose: bool = True):
    forbid_consecutive = forbid_consecutive or []

    model = pl.LpProblem("StaffScheduling", pl.LpMinimize)

    x = {(i, s): pl.LpVariable(f"x_{i}_{s}", cat=pl.LpBinary)
         for i in people for s in shifts}

    # Objective: minimize total cost
    model += pl.lpSum(cost_per_person_per_shift[i] * x[(i, s)]
                      for i in people for s in shifts), "TotalCost"

    # Coverage constraints
    for s in shifts:
        model += pl.lpSum(x[(i, s)]
                          for i in people) >= coverage_required[s], f"cov_{s}"

    # Availability constraints
    for i in people:
        for s in shifts:
            if availability.get((i, s), 0) == 0:
                model += x[(i, s)] == 0, f"avail_{i}_{s}"

    # Max shifts per person
    for i in people:
        model += pl.lpSum(x[(i, s)]
                          for s in shifts) <= max_shifts_per_person, f"max_{i}"

    # Optional: forbid certain consecutive pairs (e.g., Night->Morning)
    for (s1, s2) in forbid_consecutive:
        for i in people:
            model += x[(i, s1)] + x[(i, s2)] <= 1, f"no_consec_{i}_{s1}_{s2}"

    status = model.solve(pl.PULP_CBC_CMD(msg=False))

    # Build schedule
    schedule = {s: [] for s in shifts}
    total_cost = pl.value(model.objective)
    for s in shifts:
        for i in people:
            if x[(i, s)].value() > 0.5:
                schedule[s].append(i)

    if verbose:
        print("\n=== Workforce Scheduling Result ===")
        print("Status:", pl.LpStatus[status])
        print("Total Cost:", round(total_cost, 4))
        print("Assignments:")
        for s in shifts:
            print(
                f"  {s}: {', '.join(schedule[s]) if schedule[s] else '-'} (need {coverage_required[s]})")
        # utilization per person
        print("\nShifts per person:")
        for i in people:
            worked = sum(1 for s in shifts if x[(i, s)].value() > 0.5)
            print(f"  {i}: {worked}")

    return {
        "status": pl.LpStatus[status],
        "total_cost": total_cost,
        "schedule": schedule,
        "worked_counts": {i: sum(1 for s in shifts if x[(i, s)].value() > 0.5) for i in people}
    }

# ============================
# SAMPLE DATA & WHAT-IF KNOBS
# ============================


def sample_factory_instance():
    products = {
        "A": Product(profit=30, demand_max=40),
        "B": Product(profit=20, demand_max=10_000),
        "C": Product(profit=25, demand_max=60),
    }
    resources = {
        "MachineHrs": Resource(capacity=100),
        "MaterialKg": Resource(capacity=150),
    }
    consumption = {
        ("MachineHrs", "A"): 2,
        ("MachineHrs", "B"): 1,
        ("MachineHrs", "C"): 1.5,
        ("MaterialKg", "A"): 3,
        ("MaterialKg", "B"): 2,
        ("MaterialKg", "C"): 2.5,
    }
    return products, resources, consumption


def sample_staffing_instance():
    people = ["Alice", "Bob", "Cara", "Dee", "Evan"]
    shifts = ["Mon-Morn", "Mon-Eve", "Tue-Morn",
              "Tue-Eve", "Wed-Morn", "Wed-Eve"]

    # coverage per shift (minimum staff needed)
    coverage_required = {s: 2 for s in shifts}

    # flat cost per person per shift (could be person-specific)
    cost_per_person_per_shift = {i: c for i,
                                 c in zip(people, [90, 80, 95, 85, 70])}

    # availability matrix (1 available, 0 not available)
    availability = {}
    for i in people:
        for s in shifts:
            availability[(i, s)] = 1

    # make a few people unavailable to show constraints
    availability[("Alice", "Tue-Morn")] = 0
    availability[("Bob", "Wed-Eve")] = 0

    # Forbid working Mon-Eve then Tue-Morn ("turnaround" rule)
    forbid_consecutive = [("Mon-Eve", "Tue-Morn"), ("Tue-Eve", "Wed-Morn")]

    return people, shifts, coverage_required, cost_per_person_per_shift, availability, forbid_consecutive


# ============================
# MAIN (run both examples)
# ============================
if __name__ == "__main__":
    print("Running Factory Production Planning example...")
    products, resources, consumption = sample_factory_instance()
    result_factory = solve_production_planning(
        products, resources, consumption, verbose=True)

    print("\nRunning Workforce Scheduling example...")
    people, shifts, cov, cost, avail, no_consec = sample_staffing_instance()
    result_staff = solve_staffing(people, shifts, cov, cost, avail,
                                  max_shifts_per_person=3, forbid_consecutive=no_consec, verbose=True)

    # Quick what-if: add 10 machine hours and re-solve
    print("\nWhat-if: +10 MachineHrs capacity")
    resources["MachineHrs"].capacity += 10
    _ = solve_production_planning(
        products, resources, consumption, verbose=True)
