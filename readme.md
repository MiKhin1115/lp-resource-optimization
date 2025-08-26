# 📊 Linear Programming Resource Optimization — Streamlit App

This is a **Streamlit web app** that shows how to use **Linear Programming (LP)** and **Mixed-Integer Linear Programming (MILP)** for real-world optimization problems.

Currently, it includes two modules:

- 🏭 **Factory Production Planning**
- 👥 **Staffing Scheduler**

---

## 🚀 Features

### 🏭 Factory Module

- Decide how many units of Product A and Product B to produce.
- Constraints: limited **machine hours** and **material (kg)**.
- Objective: **maximize profit**.
- Options:
  - Solve as **LP** (fractions allowed).
  - Solve as **MILP** (whole units only).
- Shows **resource usage**, **slack**, and **shadow prices**.

---

### 👥 Staffing Module

- Assign workers to shifts at **minimum total cost**.
- Supports:
  - Per-person cost per shift.
  - Shift multipliers (e.g., night shift extra pay).
  - Min/max shifts per person.
  - Overtime threshold and multiplier.
  - Fairness weight (balances workload across staff).
- Outputs:
  - Optimal shift assignments.
  - Cost breakdown (base + overtime).
  - Warnings for **understaffing** or **unfair load distribution**.

---

## 🛠 Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – interactive UI
- [PuLP](https://coin-or.github.io/pulp/) – LP/MILP solver

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/lp-resource-optimization.git
cd lp-resource-optimization
pip install -r requirements.txt
```
