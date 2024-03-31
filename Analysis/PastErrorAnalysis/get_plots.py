import matplotlib.pyplot as plt

# Sample data for 5 parameter settings (these values are placeholders and should be replaced with your actual data)
x = [1, 5, 10, 20, 50, 100]  # Top-N values
y1 = [15.6, 61.1, 70.5, 79.0, 87.3, 91.5]  # Recall values for parameter 1
y2 = [17.8, 58.5, 65.8, 73.8, 83.3, 87.9]  # Recall values for parameter 2
y3 = [18.2, 59.1, 66.3, 75.0, 83.9, 88.8]  # Recall values for parameter 3
y4 = [17.9, 57.6, 66.3, 74.6, 82.8, 87.3]  # Recall values for parameter 4
y5 = [15.8, 50.1, 57.7, 65.3, 73.9, 79.2]  # Recall values for parameter 5

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y1, marker='o', label='Chain-of-Skills (Reproduced)')
plt.plot(x, y2, marker='o', label='Threshold:52.75 (30%)')
plt.plot(x, y3, marker='o', label='Threshold:54.61 (50%)')
plt.plot(x, y4, marker='o', label='Threshold:56.85 (70%)')
plt.plot(x, y5, marker='o', label='Threshold:60.94 (90%)')

# Annotating each point with its value
# for i in range(len(x)):
#     plt.text(x[i], y1[i], f"{y1[i]:.2f}", ha='center', va='bottom')
#     plt.text(x[i], y2[i], f"{y2[i]:.2f}", ha='center', va='bottom')
#     plt.text(x[i], y3[i], f"{y3[i]:.2f}", ha='center', va='bottom')
#     plt.text(x[i], y4[i], f"{y4[i]:.2f}", ha='center', va='bottom')
#     plt.text(x[i], y5[i], f"{y5[i]:.2f}", ha='center', va='bottom')

# Customizing the plot
plt.title('Top-k Retrieval Accuracy (Threshold)')
plt.xlabel('Top-K')
plt.ylabel('Answer Recall')
plt.xticks(x)  # Set x-ticks to be exactly at the top-N values
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('/home/shpark/COS/error_analysis/recall_comparison.png')