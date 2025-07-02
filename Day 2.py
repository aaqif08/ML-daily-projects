import pandas as pd

file_path = input("Enter the path to the iris CSV file: ")
iris_data = pd.read_csv(file_path)

print("\nSummary Statistics:")
print(iris_data.describe())

print("\nSample Count by Species:")
print(iris_data['Species'].value_counts())

filtered_data = iris_data[iris_data['PetalLengthCm'] > 1.5]
print("\nRows where PetalLengthCm > 1.5:")
print(filtered_data.to_string(index=False))

iris_data['Species_encoded'] = iris_data['Species'].astype('category').cat.codes
print("\nEncoded Species Labels:")
print(iris_data[['Species', 'Species_encoded']].drop_duplicates())

iris_data['PetalRatio'] = iris_data['PetalLengthCm'] / iris_data['PetalWidthCm']
print("\nData with new 'PetalRatio' column:")
print(iris_data[['PetalLengthCm', 'PetalWidthCm', 'PetalRatio']].head())
