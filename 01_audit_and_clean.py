import pandas as pd
import numpy as np

raw_df = pd.read_csv("AB_NYC_2019.csv")
initial_rows = len(raw_df)

# Rule 1: Drop corrupted $0/night listings (11 rows)
df = raw_df[raw_df["price"] > 0].copy()

# Rule 2: Impute structural missingness for unreviewed listings
df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

# Rule 3: Impute missing text identifiers without losing numeric metrics
df["name"] = df["name"].fillna("Unnamed Listing")
df["host_name"] = df["host_name"].fillna("Unknown Host")

# Rule 4: Segment NYC 30-day legal threshold & cap extreme minimum_nights > 365
df["rental_term"] = np.where(df["minimum_nights"] >= 30, "Long-Term (30+ Nights)", "Short-Term (<30 Nights)")
df["minimum_nights_capped"] = df["minimum_nights"].clip(upper=365)

# Rule 5: Segment price tiers, active availability & commercial hosts
df["price_tier"] = pd.cut(
    df["price"],
    bins=[0, 75, 150, 300, 10000],
    labels=["Budget ($1-$75)", "Mid-Range ($76-$150)", "Upper-Mid ($151-$300)", "Luxury ($301+)"]
)
df["is_available"] = (df["availability_365"] > 0).astype(int)
df["host_type"] = np.where(df["calculated_host_listings_count"] > 1, "Multi-Listing / Commercial", "Single-Listing Host")

# Estimated annual occupancy & revenue proxy (InsideAirbnb San Francisco model: 50% review rate, capped at 70% occupancy)
df["est_booked_nights"] = (df["reviews_per_month"] * 12 * 2 * df["minimum_nights"].clip(upper=30)).clip(upper=255)
df["est_annual_revenue"] = df["price"] * df["est_booked_nights"]

df.to_csv("nyc_airbnb_cleaned.csv", index=False)

print(f"Rows: {initial_rows:,} -> {len(df):,} (Dropped {initial_rows - len(df)} $0-price anomalies)")
print(f"Remaining Nulls (excluding expected NaT in last_review): {df.drop(columns=['last_review']).isnull().sum().sum()}")
print("\nMedian Price & Est. Annual Revenue by Borough:")
print(df.groupby("neighbourhood_group").agg(
    listings=("id", "count"),
    median_price=("price", "median"),
    mean_price=("price", "mean"),
    median_est_rev=("est_annual_revenue", "median")
).round(2))