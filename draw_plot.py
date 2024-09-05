import matplotlib.pyplot as plt

# Data for the graph
beam_sizes = [0, 1, 2, 5, 10]
recall_at_50 = [0.923, 0.931, 0.934, 0.938, 0.943]

# Creating the plot with modified axes
plt.figure(figsize=(8, 6))
plt.plot(beam_sizes, recall_at_50, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Beam Size')
plt.ylabel('Recall@50')

# Formatting y-axis to show 3 decimal places
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

# Ensuring only the beam size values appear on the x-axis
plt.xticks(beam_sizes)

plt.title('Change in Recall@50 With Varying Beam Size')

# Saving the updated graph as a PNG file
plt.savefig('./beam_size_vs_recall_at_50_v2.png')

# Show the plot for verification
plt.show()
