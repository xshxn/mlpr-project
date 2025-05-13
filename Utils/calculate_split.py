import pandas as pd


df = pd.read_csv("torgo_mainfest.csv") 

gender_counts = df["Speaker"].str.startswith("F").value_counts().rename(index={True: "Female", False: "Male"})

category_counts = df["Speaker"].value_counts()


print("Speaker Count:")
print(gender_counts)
print("\nCategory Count:")
print(category_counts)
