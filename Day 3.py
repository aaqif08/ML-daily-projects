import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_data = pd.read_csv("Iris.csv")

sns.histplot(iris_data['SepalLengthCm'], bins=10)
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()

sns.scatterplot(data=iris_data, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

sns.pairplot( iris_data,hue="Species")
plt.show()

sns.boxplot(data=iris_data, x='Species', y='PetalWidthCm')
plt.title("Boxplot of Petal Width by Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")
plt.show()
