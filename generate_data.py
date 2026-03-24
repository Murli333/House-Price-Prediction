import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "Area_sqft": np.random.randint(500, 3000, n),
    "Bedrooms": np.random.randint(1, 6, n),
    "Bathrooms": np.random.randint(1, 5, n),
    "Age": np.random.randint(0, 30, n),
    "Location_Score": np.random.randint(1, 11, n),
    "Garage": np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

df["Price"] = (
    df["Area_sqft"] * 5000 +
    df["Bedrooms"] * 300000 +
    df["Bathrooms"] * 200000 +
    df["Location_Score"] * 500000 -
    df["Age"] * 100000 +
    df["Garage"] * 250000 +
    np.random.normal(0, 200000, n)
)

df.to_csv("house_prices.csv", index=False)

print("✅ Dataset created")