import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# Define temporal parameters for the 6-fold sliding-window cross-validation
# Training uses 2023 data, testing uses the 2024 hold-out set
n_folds = 6
train_window_months = 6
val_window_months = 1
start_date = pd.to_datetime("2023-01-01") 

# Initialize the plot dimensions
fig, ax = plt.subplots(figsize=(12, 7)) 

# Generate the sliding windows for the hyperparameter tuning phase
for i in range(n_folds):
    # Calculate the dynamic start and end dates for the current fold
    train_start = start_date + pd.DateOffset(months=i)
    train_end = train_start + pd.DateOffset(months=train_window_months)
    val_end = train_end + pd.DateOffset(months=val_window_months)
    
    # Convert pandas datetimes to matplotlib numeric format for plotting
    # (AI generated)
    start_num = mdates.date2num(train_start)
    end_num = mdates.date2num(train_end)
    val_num = mdates.date2num(val_end)
    
    # Plot the training window
    # (AI generated)
    ax.barh(y=f"Fold {i+1}", 
            width=end_num - start_num, 
            left=start_num, 
            color="#4C72B0", 
            edgecolor="black",
            height=0.5,
            label="Training Data" if i == 0 else "")
    
    # Plot the validation window sequentially after the training window
    # (AI generated)
    ax.barh(y=f"Fold {i+1}", 
            width=val_num - end_num, 
            left=end_num, 
            color="#DD8452", 
            edgecolor="black",
            height=0.5,
            label="Validation Data" if i == 0 else "")

# Append the final model evaluation phase
# The finalized models are trained on the entirety of the 2023 dataset (12 months)
# and tested on the unseen 2024 dataset (12 months).
# (AI generated)
final_train_start = start_date
final_train_end = start_date + pd.DateOffset(months=train_window_months + n_folds) 
final_test_end = final_train_end + pd.DateOffset(months=12) 

final_train_start_num = mdates.date2num(final_train_start)
final_train_end_num = mdates.date2num(final_train_end)
final_test_end_num = mdates.date2num(final_test_end)

# Render a visual separator between the CV tuning phase and the final testing phase
# (AI generated)
ax.axhline(y=5.5, color="gray", linestyle="--", alpha=0.7)

# Plot the final training data block
# (AI generated)
ax.barh(y="Final\nModel", 
        width=final_train_end_num - final_train_start_num, 
        left=final_train_start_num, 
        color="#4C72B0", 
        edgecolor="black",
        height=0.5)

# Plot the 2024 hold-out test data block
# (AI generated)
ax.barh(y="Final\nModel", 
        width=final_test_end_num - final_train_end_num, 
        left=final_train_end_num, 
        color="#55A868",
        edgecolor="black",
        height=0.5,
        label="Final Test Data")

# Configure plot aesthetics and axis labels
# (AI generated)
ax.invert_yaxis()
ax.set_title("Sliding Window Validation & Final Evaluation", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Timeline", fontsize=12, fontweight="bold")
ax.set_ylabel("Cross-Validation Folds", fontsize=12, fontweight="bold")

# Format the x-axis to cleanly display the Year-Month progression
# (AI generated)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y")) 
plt.xticks(rotation=0, fontsize=9)

ax.grid(axis="x", linestyle="--", alpha=0.5)

# Align the legend below the plot to avoid overlapping with the data bars
ax.legend(loc="upper right", bbox_to_anchor=(1, -0.15), ncol=3, frameon=False)

# Strip top and right plot spines for a cleaner academic aesthetic
# (AI generated)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ensure elements are not truncated during export
plt.tight_layout()
plt.savefig("K-fold_sliding-window_Validation.pdf", format="pdf") # Export as PDF for LaTeX
plt.show()