import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from scipy import stats

df = pd.read_csv("nyc_airbnb_cleaned.csv")

# --- 1. Non-Parametric Hypothesis Tests (Skewed Marketplace Data) ---
borough_groups = [group["price"].values for _, group in df.groupby("neighbourhood_group")]
kw_stat, kw_p = stats.kruskal(*borough_groups)

st_prices = df.loc[df["rental_term"] == "Short-Term (<30 Nights)", "price"]
lt_prices = df.loc[df["rental_term"] == "Long-Term (30+ Nights)", "price"]
u_term, p_term = stats.mannwhitneyu(st_prices, lt_prices)

single_prices = df.loc[df["host_type"] == "Single-Listing Host", "price"]
multi_prices = df.loc[df["host_type"] == "Multi-Listing / Commercial", "price"]
u_host, p_host = stats.mannwhitneyu(single_prices, multi_prices)

print("=== STATISTICAL VALIDATION (NON-PARAMETRIC) ===")
print(f"Borough Price Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e}")
print(f"Rental Term Median — Short-Term (<30d): ${st_prices.median():.2f} | Long-Term (30d+): ${lt_prices.median():.2f} (p={p_term:.2e})")
print(f"Host Type Median — Single-Listing: ${single_prices.median():.2f} | Multi-Listing: ${multi_prices.median():.2f} (p={p_host:.2e})")

# --- 2. Commercial Concentration & Neighborhood Benchmarks ---
print("\n=== SUPPLY & REVENUE BY HOST TYPE ===")
print(df.groupby("host_type").agg(
    listings=("id", "count"),
    supply_share_pct=("id", lambda x: round(len(x) / len(df) * 100, 2)),
    median_price=("price", "median"),
    est_annual_rev_millions=("est_annual_revenue", lambda x: round(x.sum() / 1e6, 2))
))

nb_stats = df.groupby(["neighbourhood_group", "neighbourhood"]).agg(
    listings=("id", "count"),
    median_price=("price", "median")
).reset_index()
top10_nb = nb_stats[nb_stats["listings"] >= 100].sort_values("median_price", ascending=False).head(10)

# --- 3. 4-Panel Geospatial & Marketplace Executive Summary ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("NYC Airbnb Marketplace Hygiene, Geospatial Pricing & Host Segmentation (n = 48,884)", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Geospatial Overlay on New_York_City_.png
nyc_img = np.array(Image.open("New_York_City_.png").convert("RGB"))
# Standard InsideAirbnb NYC bounding box for New_York_City_.png
extent = [-74.258, -73.700, 40.495, 40.920]
axes[0, 0].imshow(nyc_img, extent=extent, aspect="auto", alpha=0.85)
plot_df = df[df["price"] <= 400]
sc = axes[0, 0].scatter(
    plot_df["longitude"], plot_df["latitude"],
    c=plot_df["price"], cmap="YlOrRd", alpha=0.35, s=3
)
cbar = plt.colorbar(sc, ax=axes[0, 0], fraction=0.036, pad=0.02)
cbar.set_label("Nightly Price ($ capped at $400)")
axes[0, 0].set_title("Geospatial Nightly Price Heatmap Across NYC", fontweight="bold")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")

# Panel 2: Price Distribution by Borough (<= $500 for IQR readability)
borough_order = ["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]
sns.boxplot(
    data=df[df["price"] <= 500], x="neighbourhood_group", y="price",
    order=borough_order, hue="neighbourhood_group", palette="Set2", legend=False, ax=axes[0, 1]
)
axes[0, 1].set_title("Nightly Price Distribution by Borough (<= $500 View)", fontweight="bold")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Nightly Price ($)")

# Panel 3: Top 10 Premium Neighborhoods (Min 100 Listings)
sns.barplot(
    data=top10_nb, x="median_price", y="neighbourhood",
    hue="neighbourhood_group", dodge=False, palette="deep", ax=axes[1, 0]
)
axes[1, 0].set_title("Top 10 Neighborhoods by Median Price (Min. 100 Listings)", fontweight="bold")
axes[1, 0].set_xlabel("Median Nightly Price ($)")
axes[1, 0].set_ylabel("")
axes[1, 0].legend(title="Borough", loc="lower right")

# Panel 4: Room Type Breakdown by Borough
ct_room = pd.crosstab(df["neighbourhood_group"], df["room_type"]).reindex(borough_order)
ct_room.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#1E293B", "#0EA5E9", "#F59E0B"])
axes[1, 1].set_title("Inventory Mix by Borough & Room Type", fontweight="bold")
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("Number of Listings")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].legend(title="Room Type")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("nyc_airbnb_executive_summary.png", dpi=300)
print("\nSaved chart: nyc_airbnb_executive_summary.png")