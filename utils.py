"""
Nepal E-Commerce Sales Intelligence - Utility Functions
========================================================
Reusable functions for data analysis and visualization.
Author: Sakshyam Pandit
Date: September 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from style_config import (
    CATEGORY_COLORS, CITY_COLORS, PAYMENT_COLORS,
    format_currency, format_number, add_bar_labels
)


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_and_generate_data(n_rows=1200, random_state=42):
    """
    Generate synthetic e-commerce dataset for Nepal.
    In production, replace with pd.read_csv('data/raw/sales.csv').
    
    Returns:
        pd.DataFrame with columns: OrderID, Date, City, Category, Price,
        Quantity, CustomerAge, PaymentMethod, Rating
    """
    np.random.seed(random_state)
    
    cities = ["Kathmandu", "Pokhara", "Lalitpur", "Bhaktapur", "Biratnagar", "Chitwan"]
    city_weights = [0.35, 0.18, 0.15, 0.10, 0.12, 0.10]
    
    categories = {
        "Electronics": (2500, 60000),
        "Clothing": (500, 5000),
        "Groceries": (100, 2500),
        "Home & Kitchen": (800, 15000),
        "Beauty": (300, 4000),
        "Books": (200, 3000),
    }
    
    payment_methods = ["eSewa", "Khalti", "Cash on Delivery", "Bank Transfer"]
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    
    rows = []
    for i in range(n_rows):
        cat = np.random.choice(list(categories.keys()))
        low, high = categories[cat]
        price = np.round(np.random.uniform(low, high), 2)
        qty = np.random.choice(
            [1, 1, 1, 2, 2, 3, 4, 5],
            p=[.35, .2, .15, .12, .08, .05, .03, .02]
        )
        rows.append({
            "OrderID": 1000 + i,
            "Date": np.random.choice(dates),
            "City": np.random.choice(cities, p=city_weights),
            "Category": cat,
            "Price": price,
            "Quantity": qty,
            "CustomerAge": int(np.clip(np.random.normal(30, 9), 16, 70)),
            "PaymentMethod": np.random.choice(payment_methods, p=[.35, .30, .25, .10]),
            "Rating": np.round(np.clip(np.random.normal(4.1, 0.8), 1, 5), 1),
        })
    
    df = pd.DataFrame(rows)
    
    # Inject realistic data quality issues
    for col in ["Rating", "CustomerAge", "PaymentMethod"]:
        missing_idx = np.random.choice(df.index, size=int(0.04 * n_rows), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
    
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, "Price"] = df.loc[outlier_idx, "Price"] * 25
    
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def clean_data(df):
    """
    Clean the raw dataframe:
    - Remove duplicates
    - Impute missing values
    - Cap outliers using IQR method
    
    Returns:
        pd.DataFrame: Cleaned dataframe with feature engineering
    """
    df_clean = df.drop_duplicates().copy()
    
    # Impute missing values
    df_clean["Rating"] = df_clean["Rating"].fillna(df_clean["Rating"].median())
    df_clean["CustomerAge"] = df_clean["CustomerAge"].fillna(
        df_clean["CustomerAge"].median()
    ).astype(int)
    df_clean["PaymentMethod"] = df_clean["PaymentMethod"].fillna(
        df_clean["PaymentMethod"].mode()[0]
    )
    
    # IQR-based outlier capping on Price
    Q1, Q3 = df_clean["Price"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean["Price"] = df_clean["Price"].clip(lower, upper)
    
    return df_clean


def engineer_features(df):
    """
    Create derived features for analysis.
    
    Returns:
        pd.DataFrame with additional columns: TotalAmount, Month, MonthNum,
        Weekday, AgeGroup, RevenueSegment
    """
    df = df.copy()
    
    # Core features
    df["TotalAmount"] = df["Price"] * df["Quantity"]
    df["Month"] = df["Date"].dt.month_name()
    df["MonthNum"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.day_name()
    
    # Age groups
    df["AgeGroup"] = pd.cut(
        df["CustomerAge"],
        bins=[15, 24, 34, 44, 54, 70],
        labels=["16-24", "25-34", "35-44", "45-54", "55-70"],
    )
    
    # Revenue segments for customer analysis
    revenue_bins = [0, 2000, 5000, 10000, float("inf")]
    revenue_labels = ["Budget", "Standard", "Premium", "Enterprise"]
    df["RevenueSegment"] = pd.cut(
        df["TotalAmount"], bins=revenue_bins, labels=revenue_labels
    )
    
    return df


# ============================================================================
# KPI CALCULATIONS
# ============================================================================

def calculate_kpis(df):
    """
    Calculate executive-level KPIs from the dataset.
    
    Returns:
        dict: Dictionary of KPI values
    """
    return {
        "Total Revenue": (format_currency(df["TotalAmount"].sum()), "#1a365d"),
        "Total Orders": (f"{len(df):,}", "#2c5282"),
        "Avg Order Value": (format_currency(df["TotalAmount"].mean()), "#38a169"),
        "Customer Rating": (f"{df['Rating'].mean():.1f}", "#805ad5"),
        "Unique Customers": (f"{df['OrderID'].nunique():,}", "#ed8936"),
        "Top City": (df.groupby("City")["TotalAmount"].sum().idxmax(), "#e53e3e"),
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_revenue_by_category(df, ax=None, figsize=(12, 6)):
    """Horizontal bar chart of total revenue by category."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    category_sales = df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=True)
    colors = [CATEGORY_COLORS.get(cat, "#2c5282") for cat in category_sales.index]
    
    bars = ax.barh(category_sales.index, category_sales.values, color=colors, height=0.6)
    add_bar_labels(ax, fmt="{:,.0f}")
    
    ax.set_xlabel("Total Revenue (NPR)")
    ax.set_title("Revenue by Product Category", fontsize=16, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_number(x)))
    
    return ax


def plot_revenue_by_city(df, ax=None, figsize=(12, 6)):
    """Bar chart of total revenue by city."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    city_sales = df.groupby("City")["TotalAmount"].sum().sort_values(ascending=False)
    colors = [CITY_COLORS.get(city, "#2c5282") for city in city_sales.index]
    
    bars = ax.bar(city_sales.index, city_sales.values, color=colors, width=0.6)
    add_bar_labels(ax, fmt="{:,.0f}")
    
    ax.set_ylabel("Total Revenue (NPR)")
    ax.set_xlabel("")
    ax.set_title("Revenue by City", fontsize=16, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_number(x)))
    plt.xticks(rotation=30, ha="right")
    
    return ax


def plot_monthly_trend(df, ax=None, figsize=(14, 6)):
    """Line chart with markers showing monthly revenue trend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    monthly_sales = df.groupby("MonthNum").agg(
        Revenue=("TotalAmount", "sum"),
        Orders=("OrderID", "count")
    ).reset_index()
    
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_map = {i+1: m for i, m in enumerate(month_order)}
    monthly_sales["MonthName"] = monthly_sales["MonthNum"].map(month_map)
    monthly_sales["MonthName"] = pd.Categorical(
        monthly_sales["MonthName"], categories=month_order, ordered=True
    )
    monthly_sales = monthly_sales.sort_values("MonthName")
    
    ax.plot(monthly_sales["MonthName"], monthly_sales["Revenue"],
            marker="o", linewidth=2.5, markersize=8, color="#1a365d", markerfacecolor="#ed8936")
    ax.fill_between(range(len(monthly_sales)), monthly_sales["Revenue"], alpha=0.1, color="#1a365d")
    
    ax.set_ylabel("Revenue (NPR)")
    ax.set_xlabel("")
    ax.set_title("Monthly Revenue Trend - 2024", fontsize=16, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_number(x)))
    plt.xticks(rotation=45, ha="right")
    
    return ax


def plot_payment_distribution(df, ax=None, figsize=(8, 8)):
    """Donut chart showing payment method distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    payment_counts = df["PaymentMethod"].value_counts()
    colors = [PAYMENT_COLORS.get(method, "#2c5282") for method in payment_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        payment_counts.values,
        labels=payment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=[0.03] * len(payment_counts),
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    # Add center text
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)
    ax.text(0, 0, f"Total\n{len(df):,}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1a365d")
    
    ax.set_title("Payment Method Distribution", fontsize=16, fontweight="bold", pad=20)
    
    return ax


def plot_pareto_analysis(df, ax=None, figsize=(14, 6)):
    """Pareto chart showing cumulative revenue contribution by category."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    category_sales = df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False)
    cumulative_pct = (category_sales.cumsum() / category_sales.sum()) * 100
    
    colors = [CATEGORY_COLORS.get(cat, "#2c5282") for cat in category_sales.index]
    ax.bar(category_sales.index, category_sales.values, color=colors, width=0.6)
    ax.set_ylabel("Revenue (NPR)", color="#1a365d")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_number(x)))
    
    ax2 = ax.twinx()
    ax2.plot(category_sales.index, cumulative_pct.values, color="#e53e3e",
             marker="o", linewidth=2, markersize=8)
    ax2.axhline(y=80, color="#e53e3e", linestyle="--", alpha=0.5, label="80% line")
    ax2.set_ylabel("Cumulative %", color="#e53e3e")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="center right")
    
    ax.set_title("Pareto Analysis: Revenue Contribution by Category",
                 fontsize=16, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right")
    
    return ax


def plot_revenue_heatmap(df, ax=None, figsize=(12, 6)):
    """Heatmap showing revenue by city and category."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    pivot = pd.pivot_table(
        df, values="TotalAmount", index="City", columns="Category",
        aggfunc="sum", fill_value=0
    )
    
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Revenue (NPR)"})
    ax.set_title("Revenue Heatmap: City x Category",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    
    return ax


def plot_age_distribution(df, ax=None, figsize=(12, 6)):
    """Box plot showing order value distribution by age group."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    age_order = ["16-24", "25-34", "35-44", "45-54", "55-70"]
    sns.boxplot(
        data=df, x="AgeGroup", y="TotalAmount", order=age_order,
        palette="Set2", ax=ax, width=0.5
    )
    ax.set_xlabel("Customer Age Group")
    ax.set_ylabel("Order Value (NPR)")
    ax.set_title("Order Value Distribution by Age Group",
                 fontsize=16, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_number(x)))
    
    return ax


def plot_correlation_heatmap(df, ax=None, figsize=(8, 6)):
    """Correlation heatmap for numeric variables."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    numeric_cols = ["Price", "Quantity", "CustomerAge", "Rating", "TotalAmount"]
    corr = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, ax=ax, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
    
    return ax


def plot_customer_segmentation(df, ax=None, figsize=(10, 6)):
    """Bar chart showing customer segmentation by revenue."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    segment_counts = df["RevenueSegment"].value_counts().reindex(
        ["Budget", "Standard", "Premium", "Enterprise"]
    )
    segment_colors = ["#90cdf4", "#4299e1", "#2c5282", "#1a365d"]
    
    bars = ax.bar(segment_counts.index, segment_counts.values,
                  color=segment_colors, width=0.6)
    add_bar_labels(ax, fmt="{:,.0f}")
    
    ax.set_xlabel("Revenue Segment")
    ax.set_ylabel("Number of Orders")
    ax.set_title("Customer Segmentation by Order Value",
                 fontsize=16, fontweight="bold", pad=15)
    
    return ax


def plot_monthly_growth(df, ax=None, figsize=(14, 6)):
    """Bar chart showing month-over-month revenue growth rate."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    monthly_revenue = df.groupby("MonthNum")["TotalAmount"].sum()
    growth_rate = monthly_revenue.pct_change() * 100
    
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_map = {i+1: m for i, m in enumerate(month_order)}
    
    colors = ["#38a169" if g >= 0 else "#e53e3e" for g in growth_rate.values]
    
    ax.bar(range(1, 13), growth_rate.values, color=colors, width=0.6)
    ax.axhline(y=0, color="#1a365d", linewidth=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([month_map[i][:3] for i in range(1, 13)])
    ax.set_ylabel("Growth Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Month-over-Month Revenue Growth",
                 fontsize=16, fontweight="bold", pad=15)
    
    for i, v in enumerate(growth_rate.values):
        if not np.isnan(v):
            ax.text(i + 1, v + (0.5 if v >= 0 else -1.5),
                   f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                   fontsize=8, fontweight="bold")
    
    return ax


def plot_geographic_bubble(df, ax=None, figsize=(12, 8)):
    """Bubble chart representing city-wise revenue (simulating geographic view)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    city_metrics = df.groupby("City").agg(
        Revenue=("TotalAmount", "sum"),
        Orders=("OrderID", "count"),
        AvgOrder=("TotalAmount", "mean"),
        Rating=("Rating", "mean")
    ).reset_index()
    
    # Approximate coordinates for Nepal cities (for visualization)
    city_coords = {
        "Kathmandu": (27.7172, 85.3240),
        "Pokhara": (28.2096, 83.9856),
        "Lalitpur": (27.6644, 85.3188),
        "Bhaktapur": (27.6710, 85.4298),
        "Biratnagar": (26.4525, 87.2640),
        "Chitwan": (27.5291, 84.3542),
    }
    
    city_metrics["Lat"] = city_metrics["City"].map(lambda x: city_coords[x][0])
    city_metrics["Lon"] = city_metrics["City"].map(lambda x: city_coords[x][1])
    
    scatter = ax.scatter(
        city_metrics["Lon"], city_metrics["Lat"],
        s=city_metrics["Revenue"] / city_metrics["Revenue"].max() * 2000,
        c=city_metrics["Revenue"],
        cmap="YlOrRd", alpha=0.7, edgecolors="#1a365d", linewidth=1.5
    )
    
    for _, row in city_metrics.iterrows():
        ax.annotate(
            f"{row['City']}\n{format_currency(row['Revenue'])}",
            (row["Lon"], row["Lat"]),
            fontsize=9, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
    
    plt.colorbar(scatter, ax=ax, label="Revenue (NPR)", shrink=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Revenue Distribution - Nepal",
                 fontsize=16, fontweight="bold", pad=15)
    
    return ax
