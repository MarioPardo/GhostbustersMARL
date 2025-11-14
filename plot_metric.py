import json
import matplotlib.pyplot as plt
import os

METRICS_PATH_P1 = "pymarl/results/sacred/ghostbusters_vdn/ghostbusters/"
METRICS_PATH_P2 = "/metrics.json"
OUTPUT_FIG = "metrics_summary.png"

def load_metrics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def extract_xy(data, key):
    """Return (steps, values) for a given metric key, or (None, None) if missing."""
    if key not in data:
        print(f"[Warning] Metric '{key}' not found in metrics.json, skipping.")
        return None, None

    metric = data[key]
    steps = metric.get("steps", None)
    values = metric.get("values", None)

    if steps is None or values is None:
        print(f"[Warning] Metric '{key}' does not have 'steps' or 'values', skipping.")
        return None, None

    if len(steps) != len(values):
        print(f"[Warning] Metric '{key}' has mismatched steps/values lengths, skipping.")
        return None, None

    return steps, values

def main():

    print("Plese input number of experiment you'd like to plot (integer)")
    exp_num = input().strip()
    METRICS_PATH = METRICS_PATH_P1 + exp_num + METRICS_PATH_P2

    if not os.path.exists(METRICS_PATH):
        print(f"metrics file not found: {METRICS_PATH}")
        return

    print("Input what you would like to call this analysis (string)")
    OUTPUT_FIG = input().strip() + ".png"

    data = load_metrics(METRICS_PATH)
    print("Available metrics:", list(data.keys()))

    # --- Extract the three series we care about ---
    # 1) Episode length vs env steps
    steps_ep, vals_ep = extract_xy(data, "ep_length_mean")

    # 2) Reward vs env steps
    steps_ret, vals_ret = extract_xy(data, "return_mean")

    # 3) Q-values vs env steps
    steps_q, vals_q = extract_xy(data, "q_taken_mean")

    # --- Create a single figure with 3 subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Ghostbusters MARL Training Metrics", fontsize=16)

    # Subplot 1: Episode length
    ax = axes[0]
    if steps_ep is not None:
        ax.plot(steps_ep, vals_ep)
        ax.set_ylabel("ep_length_mean")
        ax.set_title("Episode length over training")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "ep_length_mean missing", ha="center", va="center")
        ax.set_axis_off()

    # Subplot 2: Return
    ax = axes[1]
    if steps_ret is not None:
        ax.plot(steps_ret, vals_ret)
        ax.set_ylabel("return_mean")
        ax.set_title("Episode return over training")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "return_mean missing", ha="center", va="center")
        ax.set_axis_off()

    # Subplot 3: Q-values
    ax = axes[2]
    if steps_q is not None:
        ax.plot(steps_q, vals_q)
        ax.set_xlabel("env steps (t_env)")
        ax.set_ylabel("q_taken_mean")
        ax.set_title("Q-values of taken actions over training")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "q_taken_mean missing", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save & show ---
    fig.savefig(OUTPUT_FIG, dpi=150)
    print(f"Saved figure to {OUTPUT_FIG}")

    plt.show()

if __name__ == "__main__":
    main()
