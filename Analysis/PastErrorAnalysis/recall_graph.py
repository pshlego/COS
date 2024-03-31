import matplotlib.pyplot as plt

# Top-k values
top_k_values = [1, 2, 3, 4, 5]

# Recall values for each top-k. Adjust these values based on your actual data.
recall_values = [0.66, 0.73, 0.76, 0.77, 0.78]  # Example data

# Plotting the line graph
plt.figure(figsize=(10, 6))  # Set the size of the graph
plt.plot(top_k_values, recall_values, marker='o', linestyle='-', color='b')  # Plot the line graph

# Adding titles and labels
plt.title('Entity Linking Recall Variation with Increasing Top-k', fontsize=14)  # Set a more formal title
plt.xlabel('Top-k', fontsize=12)  # Set the x-axis label
plt.ylabel('Recall', fontsize=12)  # Set the y-axis label
plt.xticks(top_k_values)  # Set the values displayed on the x-axis
plt.ylim(0, 1)  # Fix the y-axis range from 0 to 1
plt.grid(True)  # Show grid

# Annotating each data point with its value
for i, value in enumerate(recall_values):
    plt.text(top_k_values[i], value + 0.02, f'{value:.2f}', ha='center')  # Adjust text position and format

# Save the graph as a PNG file
plt.savefig('/home/shpark/COS/error_analysis/entity_linking_recall.png', format='png', dpi=300)  # Save the figure
