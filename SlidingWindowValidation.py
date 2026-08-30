import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# 1. Configuration
n_folds = 6
train_window_months = 6
val_window_months = 1
start_date = pd.to_datetime("2023-01-01") # Start of Year 1

# 2. Setup the Plot
fig, ax = plt.subplots(figsize=(12, 7)) # Slightly taller to fit the legend below

# 3. Loop through and draw each fold
for i in range(n_folds):
    # Calculate dates for the current fold
    train_start = start_date + pd.DateOffset(months=i)
    train_end = train_start + pd.DateOffset(months=train_window_months)
    val_end = train_end + pd.DateOffset(months=val_window_months)
    
    # Convert dates to matplotlib numbers
    start_num = mdates.date2num(train_start)
    end_num = mdates.date2num(train_end)
    val_num = mdates.date2num(val_end)
    
    # Plot Training Window (Blue)
    ax.barh(y=f"Fold {i+1}", 
            width=end_num - start_num, 
            left=start_num, 
            color="#4C72B0", 
            edgecolor="black",
            height=0.5,
            label="Training Data" if i == 0 else "")
    
    # Plot Validation Window (Orange)
    ax.barh(y=f"Fold {i+1}", 
            width=val_num - end_num, 
            left=end_num, 
            color="#DD8452", 
            edgecolor="black",
            height=0.5,
            label="Validation Data" if i == 0 else "")

# 4. NEW: Add Final Testing Phase
# The final model trains on all 18 months of data used during tuning
final_train_start = start_date
final_train_end = start_date + pd.DateOffset(months=train_window_months + n_folds) 
final_test_end = final_train_end + pd.DateOffset(months=12) # The remaining 6 months of Year 2

final_train_start_num = mdates.date2num(final_train_start)
final_train_end_num = mdates.date2num(final_train_end)
final_test_end_num = mdates.date2num(final_test_end)

# Draw a dashed line to separate tuning from final testing
ax.axhline(y=5.5, color='gray', linestyle='--', alpha=0.7)

# Plot Final Training Data (Blue)
ax.barh(y="Final\nModel", 
        width=final_train_end_num - final_train_start_num, 
        left=final_train_start_num, 
        color="#4C72B0", 
        edgecolor="black",
        height=0.5)

# Plot Final Test Data (Green)
ax.barh(y="Final\nModel", 
        width=final_test_end_num - final_train_end_num, 
        left=final_train_end_num, 
        color="#55A868",  # Green color for unseen test data
        edgecolor="black",
        height=0.5,
        label="Final Test Data")

# 5. Formatting the Plot
ax.invert_yaxis() # Put Fold 1 at the top
ax.set_title("Sliding Window Validation & Final Evaluation", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Timeline", fontsize=12, fontweight='bold')
ax.set_ylabel("Cross-Validation Folds", fontsize=12, fontweight='bold')

# Format x-axis to show Year-Month cleanly
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y")) 
plt.xticks(rotation=0, fontsize=9)

# Add gridlines 
ax.grid(axis='x', linestyle='--', alpha=0.5)

# NEW: Legend placed below the plot, aligned to the right
ax.legend(loc="upper right", bbox_to_anchor=(1, -0.15), ncol=3, frameon=False)

# Remove top and right borders for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Use bbox_inches="tight" to prevent the bottom legend from being cut off
plt.tight_layout()
plt.savefig("K-fold sliding-window Validation")
plt.show()