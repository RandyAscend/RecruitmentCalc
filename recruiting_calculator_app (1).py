import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recruiting Calculator", layout="wide")

st.title("Recruiting Pipeline Calculator")

# Fixed stage order
STAGES = [
    "Screening call",
    "Technical screen",
    "Take-home review",
    "Onsite",
    "Offer",
    "Hired",
]

st.caption("Enter your assumptions below. No data files needed.")

# ============= Controls =============
left, right = st.columns([2, 1])

with left:
    target_type = st.selectbox("Target outcome", ["Hires", "Offers"], index=0)
    target_value = st.number_input(f"Number of {target_type.lower()} you want", min_value=1, value=3, step=1)

    timeframe_days = st.number_input("Timeframe (days)", min_value=1, value=30, step=1)
    workdays_per_week = st.number_input("Working days per week", min_value=1, max_value=7, value=5, step=1)
    screens_per_day = st.number_input("Recruiter screens per working day", min_value=1, value=10, step=1)

with right:
    st.write("Existing candidates already in pipeline")
    inventory = {}
    for s in STAGES[:-1]:  # no inventory input for terminal stage
        inventory[s] = st.number_input(f"In '{s}'", min_value=0, value=0, step=1, key=f"inv_{s}")

st.markdown("---")

st.subheader("Stage assumptions")

cols = st.columns(3)

# Pass rates: per transition to the next stage
with cols[0]:
    st.markdown("**Pass rate to next stage**")
    pass_rates = []
    for i, s in enumerate(STAGES[:-1]):
        default_pr = 0.6 if i == 0 else 0.5
        pr = st.slider(f"{s} → {STAGES[i+1]}", 0.0, 1.0, float(default_pr), 0.01, key=f"pr_{s}")
        pass_rates.append(pr)

# Avg days per stage
with cols[1]:
    st.markdown("**Average days in stage**")
    avg_days = []
    for s in STAGES[:-1]:  # terminal stage has no days
        d = st.number_input(f"{s}", min_value=0.0, value=3.0, step=0.5, key=f"days_{s}")
        avg_days.append(d)

# Which stage is stage 0 for counting "screens"
with cols[2]:
    st.markdown("**Stage 0 (where you start counting screens)**")
    stage0 = st.selectbox("Choose stage 0", STAGES[:-1], index=0)

# Helpers
def subset_chain(stages, target_type, stage0):
    i0 = stages.index(stage0)
    end_token = "offer" if target_type == "Offers" else "hired"
    end_idx = None
    for i, s in enumerate(stages):
        if end_token in s.lower():
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(stages) - 1
    i0 = max(0, min(i0, len(stages) - 2))
    end_idx = max(i0 + 1, min(end_idx, len(stages) - 1))
    return stages[i0:end_idx+1], i0, end_idx

effective_stages, i0, i_end = subset_chain(STAGES, target_type, stage0)

eff_pass = pass_rates[i0:i_end]  # transitions
eff_days = avg_days[i0:i_end] if i_end > i0 else []

def forward_counts(start_n, pass_list):
    counts = [start_n]
    for pr in pass_list:
        counts.append(int(np.floor(counts[-1] * pr)))
    return counts

def expected_from_inventory(inv_dict, stages, global_stages, pass_rates):
    total = 0.0
    for i, s in enumerate(stages[:-1]):
        inv = float(inv_dict.get(s, 0))
        if inv <= 0:
            continue
        gi = global_stages.index(s)
        ge = global_stages.index(stages[-1])
        pr_slice = pass_rates[gi:ge]
        conv = inv * (np.prod(pr_slice) if pr_slice else 1.0)
        total += conv
    return total

# Compute inventory-based projected outcome first
inventory_contrib = expected_from_inventory(inventory, effective_stages, STAGES, pass_rates)
inventory_proj = int(np.floor(inventory_contrib))

# Chain probability from stage0 to target
p_chain = float(np.prod(eff_pass)) if eff_pass else 1.0

# How many additional screens are needed beyond inventory to reach target?
net_needed = max(0, target_value - inventory_proj)
required_from_stage0 = int(np.ceil(net_needed / max(p_chain, 1e-9))) if net_needed > 0 else 0

# Projected new outcome from those required screens
new_counts_chain = forward_counts(required_from_stage0, eff_pass)
new_proj = new_counts_chain[-1] if new_counts_chain else 0

# Totals
projected_total = inventory_proj + new_proj

# Capacity
workdays_available = int(np.ceil((timeframe_days / 7.0) * workdays_per_week))
capacity_screens = workdays_available * int(screens_per_day)

# Cycle time estimate
total_cycle_days = sum(eff_days)

st.markdown("---")
st.subheader("Results")

c1, c2, c3 = st.columns(3)
c1.metric(f"Required {stage0} count", required_from_stage0)
c2.metric(f"Projected {target_type.lower()} (inventory + new)", projected_total)
c3.metric("Estimated cycle time (days)", f"{total_cycle_days:.0f}")

# Feasibility text
if required_from_stage0 <= capacity_screens:
    st.success(f"Feasible within capacity: {capacity_screens} {stage0.lower()}s in {workdays_available} working days")
else:
    st.error(f"Not feasible with current capacity. Max {stage0.lower()}s: {capacity_screens} in {workdays_available} working days")

# Breakdown table
breakdown = pd.DataFrame({
    "Stage": effective_stages,
    "Avg Days": eff_days + ([np.nan] if len(eff_days) < len(effective_stages) else []),
    "Pass Rate to Next": eff_pass + [np.nan] if eff_pass else [np.nan],
    "Projected New Count": new_counts_chain if new_counts_chain else [],
})

st.write("Projected counts by stage from **new** candidates only:")
st.dataframe(breakdown)

# Inventory summary
st.write(f"Expected {target_type.lower()} from **current inventory**: {inventory_proj}")
st.write(f"Expected {target_type.lower()} from **new** {stage0.lower()}s: {new_proj}")
st.write(f"**Total projected {target_type.lower()}**: {projected_total}")

# Download scenario
combined = breakdown.copy()
combined.loc[len(combined)] = ["<TOTALS>", np.nan, np.nan, projected_total]
st.download_button(
    "Download scenario as CSV",
    combined.to_csv(index=False),
    file_name="recruiting_calculator_scenario.csv",
    mime="text/csv"
)
