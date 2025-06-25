import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Data Preparation ---
# Original metrics from the paper
original_scientific = {
    "NDCG@10": 0.1027, "Recall@10": 0.1448, "MRR": 0.0951
}
original_instruments = {
    "NDCG@10": 0.0830, "Recall@10": 0.1052, "MRR": 0.0807
}

# Reproduction metrics
repro_scientific = {
    "NDCG@10": 0.1046, "Recall@10": 0.1460, "MRR": 0.0974
}
repro_instruments = {
    "NDCG@10": 0.0832, "Recall@10": 0.1048, "MRR": 0.08096
}

# --- Plotting ---

# Set a professional and clean style
sns.set_style("whitegrid")

# Create a figure with two subplots, side-by-side, with a shared Y-axis
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle('Comparison of Original vs. Reproduction Metrics Across Datasets', fontsize=20, y=0.98)

# *** FIX: Calculate the global maximum score across all data ***
all_scores_combined = (
    list(original_scientific.values()) + list(repro_scientific.values()) +
    list(original_instruments.values()) + list(repro_instruments.values())
)
global_max_score = max(all_scores_combined)

# A dictionary to loop through our data and titles
datasets = {
    "Amazon Scientific Dataset": (axes[0], original_scientific, repro_scientific),
    "Amazon Instruments Dataset": (axes[1], original_instruments, repro_instruments)
}

# --- Loop to populate each subplot ---
for title, (ax, original_data, repro_data) in datasets.items():
    metrics = list(original_data.keys())
    original_scores = list(original_data.values())
    repro_scores = list(repro_data.values())
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    # Plotting the bars for original and reproduction metrics
    rects1 = ax.bar(x - width/2, original_scores, width, label='Original Paper', 
                    color=sns.color_palette("Blues")[3])
    rects2 = ax.bar(x + width/2, repro_scores, width, label='Reproduction', 
                    color=sns.color_palette("Oranges")[3])

    # Add text for labels, title, and ticks
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, rotation=10)
    ax.tick_params(axis='y', labelsize=11)
    
    # Function to attach a text label above each bar
    def autolabel(rects, ax_to_use):
        for rect in rects:
            height = rect.get_height()
            ax_to_use.annotate(f'{height:.4f}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)

    autolabel(rects1, ax)
    autolabel(rects2, ax)

# *** FIX: Apply the calculated Y-limit to both subplots ***
# We add a 20% buffer on top for the labels
y_limit_top = global_max_score * 1.20
axes[0].set_ylim(0, y_limit_top)

# Set a shared Y-label for the entire figure
axes[0].set_ylabel('Metric Score', fontsize=14)

# Create a single, shared legend for the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=12)

# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust rect to make space for the suptitle
plt.show()