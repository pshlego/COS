import matplotlib.pyplot as plt

# Step 2: Define the top-k values and the corresponding recall values
top_k = [1, 2, 3, 4, 5]
recall_values = [0.843, 0.882, 0.899, 0.905, 0.914]  # Example recall values for gold passages

# Step 3: Define the recall value for the baseline model
baseline_recall = 0.884  # Example recall value for the baseline model

# Step 4: Create the plot
plt.figure(figsize=(10, 6))
plt.plot(top_k, recall_values, marker='o', linestyle='-', color='blue', label='Answer Recall of the Entity Linking Module')

# Step 5: Add the red horizontal line for the baseline model's recall
plt.axhline(y=baseline_recall, color='red', linestyle='--', label='Answer Recall of the Expanded Query Retrieval Module')

# Set the y-axis limit from 0 to 1
plt.ylim(0.6, 1)

# Display each recall value on the plot
for i, txt in enumerate(recall_values):
    plt.annotate(txt, (top_k[i], recall_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.annotate(f'{baseline_recall:.3f}', xy=(max(top_k)/2, baseline_recall), textcoords="offset points", xytext=(0,-15), ha='center', color='red')

# Step 6: Customize the plot
plt.title('Answer Recall (passage only)')
plt.xlabel('Top-K')
plt.ylabel('Recall')
plt.xticks(top_k)
plt.legend()
plt.grid(True)
plt.savefig('/home/shpark/COS/error_analysis/results_all/recall_plot.png')
