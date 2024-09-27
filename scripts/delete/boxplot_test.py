import seaborn as sns
import matplotlib.pyplot as plt

# Example data
data = sns.load_dataset("iris")

# Create a boxplot with notches
sns.boxplot(x="species", y="sepal_length", data=data, notch=True)

# Show the plot
plt.show()
