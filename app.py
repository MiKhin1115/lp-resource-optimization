# Run with: streamlit run app.py
# Requirements: pip install streamlit pulp
# Easy app with clear warnings + correct staffing cost (supports shift differentials)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import pulp as pl

st.set_page_config(
    page_title="LP Resource Optimization — Easy App", layout="centered")

# =====================================
# Data classes (clarity only)
# =====================================


@dataclass
class Product:
    profit: float
    demand_max: float


@dataclass
class Resource:
    capacity: float

# =====================================
# Helper: nice status panel
# =====================================


def show_solver_status(status_str: str):
    if status_str == "Optimal":
        st.success("Optimal solution found ✅")
    elif status_str == "Infeasible":
        st.error(
            "No feasible solution exists. Please relax capacities, increase availability, or lower requirements.")
    elif status_str == "Unbounded":
        st.error(
            "The model is unbounded. Add realistic upper limits (e.g., demand caps or capacity bounds).")
    else:
        st.warning(
            f"Solver finished with status: {status_str}. Consider revising inputs or trying again.")

# =====================================
# Factory model (LP)
# =====================================


def solve_production_planning(products: Dict[str, Product],
                              resources: Dict[str, Resource],
                              consumption: Dict[Tuple[str, str], float]):
    model = pl.LpProblem("ProductionPlanning", pl.LpMaximize)

    # Decision variables
    x = {p: pl.LpVariable(f"x_{p}", lowBound=0) for p in products}

    # Objective: profit = sum(p_i * x_i)
    model += pl.lpSum(products[p].profit * x[p]
                      for p in products), "TotalProfit"

    # Resource capacity constraints
    cons_resource = {}
    for r in resources:
        cname = f"cap_{r}"
        model += pl.lpSum(consumption.get((r, p), 0.0) *
                          x[p] for p in products) <= resources[r].capacity, cname
        cons_resource[r] = model.constraints[cname]

    # Demand limits
    for p in products:
        model += x[p] <= products[p].demand_max, f"dem_{p}"

    # Solve
    status_code = model.solve(pl.PULP_CBC_CMD(msg=False))
    status_str = pl.LpStatus[status_code]

    # Results container
    values = {p: float(x[p].value() or 0.0) for p in products}
    objective = float(pl.value(model.objective) or 0.0)

    # Slack, shadow prices
    resource_rows = []
    for r, c in cons_resource.items():
        used = sum(consumption.get((r, p), 0.0) * values[p] for p in products)
        slack = float(getattr(c, "slack", 0.0))
        dual = getattr(c, "pi", None)
        dual = None if dual is None else float(dual)
        resource_rows.append({
            "resource": r,
            "used": used,
            "capacity": float(resources[r].capacity),
            "slack": slack,
            "shadow_price": dual,
        })

    # Reduced costs (for multiple-optima hint)
    reduced_costs = {}
    for p in products:
        try:
            dj = getattr(x[p], "dj")
            reduced_costs[p] = None if dj is None else float(dj)
        except Exception:
            reduced_costs[p] = None

    return {
        "status": status_str,
        "objective": objective,
        "x": values,
        "resource_rows": resource_rows,
        "reduced_costs": reduced_costs,
    }

# =====================================
# Staffing model (MILP) with shift differentials, overtime, and fairness


def solve_staffing(people: List[str],
                   shifts: List[str],
                   coverage_required: Dict[str, int],
                   base_cost_per_person: Dict[str, float],
                   shift_multiplier: Dict[str, float],
                   availability: Dict[Tuple[str, str], int],
                   min_shifts_per_person: int,
                   max_shifts_per_person: int,
                   overtime_threshold: int,
                   overtime_multiplier: float,
                   fairness_weight: float = 0.0):
    """Solve staffing with:
    - per-person base cost × per-shift multiplier
    - min/max shifts fairness constraints
    - overtime: surcharge beyond threshold (linear approximation)
    - optional fairness objective (L1 distance to target)
    """
    model = pl.LpProblem("StaffScheduling", pl.LpMinimize)

    # Decision: assign or not
    x = {(i, s): pl.LpVariable(f"x_{i}_{s}", cat=pl.LpBinary)
         for i in people for s in shifts}

    # Workload per person
    w = {i: pl.lpSum(x[(i, s)] for s in shifts) for i in people}

    # Overtime (continuous; will match integer gap at optimum)
    o = {i: pl.LpVariable(f"o_{i}", lowBound=0) for i in people}

    # Fairness deviation variables around target
    total_required = sum(coverage_required[s] for s in shifts)
    target = total_required / max(1, len(people))
    d_plus = {i: pl.LpVariable(f"dplus_{i}", lowBound=0) for i in people}
    d_minus = {i: pl.LpVariable(f"dminus_{i}", lowBound=0) for i in people}

    # Objective components
    # Base staffing cost with shift differentials
    base_cost_expr = pl.lpSum(base_cost_per_person[i] * shift_multiplier[s] * x[(i, s)]
                              for i in people for s in shifts)

    # Overtime surcharge (approx): surcharge per extra shift = base_cost[i] * avg_mult * (ot_mult - 1)
    avg_mult = sum(shift_multiplier[s] for s in shifts) / max(1, len(shifts))
    ot_surcharge = pl.lpSum(base_cost_per_person[i] * avg_mult * (overtime_multiplier - 1.0) * o[i]
                            for i in people)

    # Fairness penalty (L1): weight * sum |w_i - target|
    fairness_penalty = fairness_weight * \
        pl.lpSum(d_plus[i] + d_minus[i] for i in people)

    model += base_cost_expr + ot_surcharge + \
        fairness_penalty, "TotalCostWithPolicies"

    # Coverage per shift
    for s in shifts:
        model += pl.lpSum(x[(i, s)]
                          for i in people) >= coverage_required[s], f"cov_{s}"

    # Availability
    for i in people:
        for s in shifts:
            if availability.get((i, s), 0) == 0:
                model += x[(i, s)] == 0, f"avail_{i}_{s}"

    # Min/Max shifts fairness constraints
    for i in people:
        model += w[i] <= max_shifts_per_person, f"max_{i}"
        model += w[i] >= min_shifts_per_person, f"min_{i}"

    # Overtime definition: o_i >= w_i - T
    for i in people:
        model += o[i] >= w[i] - overtime_threshold, f"ot_lb_{i}"
        # Not needed: o_i >= 0 already

    # Fairness balance: w_i - target = dplus_i - dminus_i
    for i in people:
        model += w[i] - target == d_plus[i] - d_minus[i], f"fair_{i}"

    status_code = model.solve(pl.PULP_CBC_CMD(msg=False))
    status_str = pl.LpStatus[status_code]

    # Read solution
    schedule = {s: [] for s in shifts}
    for s in shifts:
        for i in people:
            val = x[(i, s)].value()
            if (val or 0) > 0.5:
                schedule[s].append(i)

    worked_counts = {i: int(round(sum(1 for s in shifts if (
        x[(i, s)].value() or 0) > 0.5))) for i in people}
    overtime_counts = {i: float(o[i].value() or 0.0) for i in people}

    total_cost = float(pl.value(model.objective) or 0.0)

    return {
        "status": status_str,
        "total_cost": total_cost,
        "schedule": schedule,
        "worked_counts": worked_counts,
        "overtime_counts": overtime_counts,
        "target": target,
        "avg_multiplier": avg_mult,
    }

# =====================================
# UI
# =====================================


st.title("Linear Programming for Resource Optimization — Easy App")
st.markdown(
    "Pick a module → change a few numbers → **Optimize**."


    "The app warns about: **infeasible**, **unbounded**, **fractional outputs**, **tight constraints**, and understaffing.")

module = st.radio("Choose a module:", [
                  "Factory (maximize profit)", "Staffing (minimize wage cost)"])

# -------------------------------------
# FACTORY MODULE
# -------------------------------------
if module == "Factory (maximize profit)":
    st.header("Factory: 2 products × 2 resources")
    with st.expander("About", expanded=False):
        st.write(
            "We decide how many units of A and B to make. Limited Machine Hours and Material. Goal: maximize profit.")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Products")
        pA = st.number_input("Profit of A", value=30.0)
        dA = st.number_input("Max demand of A", value=40.0)
        pB = st.number_input("Profit of B", value=20.0)
        dB = st.number_input(
            "Max demand of B (big number = unlimited)", value=10000.0)
    with colB:
        st.subheader("Resources")
        capM = st.number_input("Machine hours capacity", value=100.0)
        capR = st.number_input("Material (kg) capacity", value=150.0)

    st.subheader("Resource use per unit")
    c1, c2 = st.columns(2)
    with c1:
        mA = st.number_input("Machine hours per A", value=2.0)
        mB = st.number_input("Machine hours per B", value=1.0)
    with c2:
        rA = st.number_input("Material kg per A", value=3.0)
        rB = st.number_input("Material kg per B", value=2.0)

    # Toggle to require integers (demo purpose only)
    need_integers = st.checkbox("Require whole units (convert to MILP)",
                                value=False, help="If on, A and B must be integers.")

    if st.button("Optimize factory plan", type="primary"):
        # Build data
        products = {"A": Product(pA, dA), "B": Product(pB, dB)}
        resources = {"MachineHrs": Resource(
            capM), "MaterialKg": Resource(capR)}
        consumption = {
            ("MachineHrs", "A"): mA, ("MachineHrs", "B"): mB,
            ("MaterialKg", "A"): rA, ("MaterialKg", "B"): rB,
        }

        # Solve (LP or MILP if integer toggle)
        if not need_integers:
            res = solve_production_planning(products, resources, consumption)
        else:
            # Simple MILP version enforcing integers
            model = pl.LpProblem("ProductionPlanningInt", pl.LpMaximize)
            xA = pl.LpVariable("x_A", lowBound=0, cat=pl.LpInteger)
            xB = pl.LpVariable("x_B", lowBound=0, cat=pl.LpInteger)
            model += pA * xA + pB * xB
            model += mA * xA + mB * xB <= capM
            model += rA * xA + rB * xB <= capR
            model += xA <= dA
            model += xB <= dB
            status_code = model.solve(pl.PULP_CBC_CMD(msg=False))
            status_str = pl.LpStatus[status_code]
            res = {
                "status": status_str,
                "objective": float(pl.value(model.objective) or 0.0),
                "x": {"A": float(xA.value() or 0.0), "B": float(xB.value() or 0.0)},
                "resource_rows": [
                    {"resource": "MachineHrs", "used": float(mA*(xA.value() or 0)+mB*(xB.value() or 0)), "capacity": float(
                        capM), "slack": float(capM - (mA*(xA.value() or 0)+mB*(xB.value() or 0))), "shadow_price": None},
                    {"resource": "MaterialKg", "used": float(rA*(xA.value() or 0)+rB*(xB.value() or 0)), "capacity": float(
                        capR), "slack": float(capR - (rA*(xA.value() or 0)+rB*(xB.value() or 0))), "shadow_price": None},
                ],
                "reduced_costs": {"A": None, "B": None},
            }

        # 1) Status
        show_solver_status(res["status"])

        # 2) Display results if any
        if res["status"] == "Optimal":
            st.write("**Optimal quantities:**")
            st.write({k: round(v, 4) for k, v in res["x"].items()})
            st.info(f"Optimal Profit: {res['objective']:.2f}")

            # 3) Warnings: fractional outputs (LP only)
            if not need_integers:
                fractional = [k for k, v in res["x"].items()
                              if abs(v - round(v)) > 1e-6]
                if fractional:
                    st.warning(
                        "Fractions detected (e.g., 7.3 units). Turn on 'Require whole units' to use integers (MILP).")

            # 4) Resource usage + tightness warning
            all_binding = True
            st.write("**Resource usage** (shadow price = value of +1 extra unit):")
            for row in res["resource_rows"]:
                st.write(
                    f"- {row['resource']}: used {row['used']:.2f} / {row['capacity']} · slack {row['slack']:.2f} · shadow price {row['shadow_price']}")
                if abs(row["slack"]) > 1e-7:
                    all_binding = False
            if all_binding and res["resource_rows"]:
                st.warning(
                    "All resources are binding (slack≈0). Consider increasing capacity to improve profit.")

            # 5) Possible multiple optima hint
            near_zero_rc = []
            for p, dj in res["reduced_costs"].items():
                if dj is not None and abs(dj) < 1e-8:
                    near_zero_rc.append(p)
            if len(near_zero_rc) > 1:
                st.info(
                    "Multiple optimal plans may exist. This is one of many equally good solutions.")

# -------------------------------------
# STAFFING MODULE
# -------------------------------------
else:
    st.header("Staffing: 4 people × 4 shifts")
    with st.expander("About", expanded=False):
        st.write("Assign people to shifts to meet coverage at minimum wage cost. Supports: shift differentials, min/max shifts, overtime (beyond a threshold), and optional fairness to balance workload.")

    people = ["Alice", "Bob", "Cara", "Dee"]
    shifts = ["Mon-AM", "Mon-PM", "Tue-AM", "Tue-PM"]

    st.subheader("Per-person base cost per shift")
    ca, cb, cc, cd = st.columns(4)
    with ca:
        costA = st.number_input("Alice", value=90.0)
    with cb:
        costB = st.number_input("Bob", value=80.0)
    with cc:
        costC = st.number_input("Cara", value=95.0)
    with cd:
        costD = st.number_input("Dee", value=85.0)
    base_cost = {"Alice": costA, "Bob": costB, "Cara": costC, "Dee": costD}

    st.subheader("Shift cost multiplier (e.g., night diff)")
    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        mult1 = st.number_input("Mon-AM ×", value=1.00)
    with sm2:
        mult2 = st.number_input("Mon-PM ×", value=1.10)
    with sm3:
        mult3 = st.number_input("Tue-AM ×", value=1.00)
    with sm4:
        mult4 = st.number_input("Tue-PM ×", value=1.10)
    shift_mult = {"Mon-AM": mult1, "Mon-PM": mult2,
                  "Tue-AM": mult3, "Tue-PM": mult4}

    st.subheader("Required staff per shift")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        req1 = st.number_input("Mon-AM", min_value=0, value=2, step=1)
    with r2:
        req2 = st.number_input("Mon-PM", min_value=0, value=2, step=1)
    with r3:
        req3 = st.number_input("Tue-AM", min_value=0, value=2, step=1)
    with r4:
        req4 = st.number_input("Tue-PM", min_value=0, value=2, step=1)
    coverage_required = {"Mon-AM": req1,
                         "Mon-PM": req2, "Tue-AM": req3, "Tue-PM": req4}

    st.subheader("Availability (tick if available)")
    availability = {}
    grid = st.columns(len(shifts) + 1)
    grid[0].markdown("**Person**")
    for j, s in enumerate(shifts):
        grid[j+1].markdown(f"**{s}**")

    for i, person in enumerate(people):
        row = st.columns(len(shifts) + 1)
        row[0].write(person)
        for j, s in enumerate(shifts):
            default = not (person == "Alice" and s ==
                           "Tue-AM") and not (person == "Bob" and s == "Tue-PM")
            availability[(person, s)] = 1 if row[j+1].checkbox("",
                                                               value=default, key=f"chk_{person}_{s}") else 0

    st.subheader("Rules & policy")
    colx, coly = st.columns(2)
    with colx:
        min_shifts = st.number_input(
            "Min shifts per person", min_value=0, value=0, step=1)
        max_shifts = st.number_input(
            "Max shifts per person", min_value=1, value=3, step=1)
    with coly:
        ot_threshold = st.number_input(
            "Overtime threshold (shifts)", min_value=0, value=3, step=1)
        ot_multiplier = st.number_input(
            "Overtime pay multiplier", min_value=1.0, value=1.5, step=0.1)

    st.subheader("Fairness (optional)")
    fair_col1, fair_col2 = st.columns(2)
    with fair_col1:
        use_fairness = st.checkbox(
            "Balance workload around average", value=True)
    with fair_col2:
        fairness_weight = st.number_input(
            "Fairness weight (0 = off)", min_value=0.0, value=5.0 if use_fairness else 0.0, step=1.0)
        if not use_fairness:
            fairness_weight = 0.0

    if st.button("Optimize staffing", type="primary"):
        res = solve_staffing(
            people, shifts, coverage_required,
            base_cost, shift_mult, availability,
            int(min_shifts), int(max_shifts), int(ot_threshold), float(
                ot_multiplier), float(fairness_weight)
        )

        # 1) Status
        show_solver_status(res["status"])

        if res["status"] == "Optimal":
            # Cost breakdown: recompute from assignments (base × multiplier) + OT surcharge summary
            rows = []
            base_sum = 0.0
            for s in shifts:
                for i in res["schedule"][s]:
                    line_cost = float(base_cost[i]) * float(shift_mult[s])
                    rows.append(
                        {"person": i, "shift": s, "base_cost": base_cost[i], "shift_mult": shift_mult[s], "line_cost": line_cost})
                    base_sum += line_cost

            st.info(f"Total Cost (objective): {res['total_cost']:.2f}")
            st.write(
                "**Base cost breakdown:** (each assignment = base × multiplier)")
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.write("No assignments (check availability/requirements).")

            # OT summary (approximate surcharge part shown)
            avg_mult = res.get("avg_multiplier", 1.0)
            ot_surcharge_sum = sum(
                base_cost[i] * avg_mult * (ot_multiplier - 1.0) * res["overtime_counts"][i] for i in people)
            st.write(
                f"**Overtime summary:** threshold={ot_threshold}, multiplier={ot_multiplier}×, average shift multiplier≈{avg_mult:.2f}")
            st.write(
                {i: f"{res['overtime_counts'][i]:.2f} extra shifts" for i in people})
            st.write(f"Approx. OT surcharge added: {ot_surcharge_sum:.2f}")

            # Assignments and load
            st.write("**Assignments per shift:**")
            understaffed = []
            for s in shifts:
                crew = res["schedule"][s]
                st.write(
                    f"- {s}: {', '.join(crew) if crew else '-'} (need {coverage_required[s]})")
                if len(crew) < coverage_required[s]:
                    understaffed.append(s)

            st.write("**Shifts per person:**")
            for person, cnt in res["worked_counts"].items():
                st.write(f"- {person}: {cnt}")

            # Tightness warnings
            if understaffed:
                st.warning("Coverage too tight on: " + ", ".join(understaffed) +
                           ". Increase availability, add people, or reduce requirements.")

            # Fairness hint
            loads = list(res["worked_counts"].values())
            if loads:
                max_load, min_load = max(loads), min(loads)
                if max_load - min_load >= 2:
                    st.info(
                        "Noticeable load imbalance. Consider stronger fairness weight or tighter min/max shifts.")

st.caption("Change values → Optimize. Now with overtime and fairness controls. Transparent breakdowns included.")
