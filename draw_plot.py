import matplotlib.pyplot as plt

# Data for the graph
beam_sizes = [0, 1, 2, 5, 10]
recall_at_50 = [92.8, 93.6, 93.9, 94.2, 94.7]

# Creating the plot with modified axes
plt.figure(figsize=(8, 6))

# Plot with improved styling
plt.plot(beam_sizes, recall_at_50, marker='o', linestyle='-', color='#1f77b4', markersize=10, linewidth=2)

# Adding labels and title
plt.xlabel('Beam Size', fontsize=25)
plt.ylabel('AR@50', fontsize=25)

# Formatting y-axis to show 3 decimal places
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

# Ensuring only the beam size values appear on the x-axis
plt.xticks(beam_sizes, fontsize=20)
plt.yticks(fontsize=20)

# Adding gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adding subtle background color for aesthetics
plt.gca().set_facecolor('#f9f9f9')

# Tight layout to prevent cutting off labels
plt.tight_layout()

# Saving the updated graph as a PNG file
plt.savefig('/home/shpark/OTT_QA_Workspace/beam_size_sensitivity.pdf', dpi=300)