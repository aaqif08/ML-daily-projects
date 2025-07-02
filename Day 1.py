import pandas as pd

file = input("Enter the path to the iris CSV file: ")
iris_data = pd.read_csv(file)

if iris_data.isnull().values.any():
    print("\nColumns with null values:")
    for col in iris_data.columns:
        if iris_data[col].isnull().any():
            print(f"{col}: {iris_data[col].isnull().sum()} null value(s)")

    print("\nRows with null values:")
    print(iris_data[iris_data.isnull().any(axis=1)].to_string(index=False))
else:
    print("\nNo null values found in the dataset.")
