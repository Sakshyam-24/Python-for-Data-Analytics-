"""
Nepal E-Commerce Sales Intelligence - Style Configuration
=========================================================
Consistent styling for all notebooks in this project.
Author: Sakshyam Pandit
Date: September 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ============================================================================
# COLOR PALETTE
# ============================================================================
PROFESSIONAL_PALETTE = {
    "primary": "#1a365d",      # Dark Navy
    "secondary": "#2c5282",    # Medium Blue
    "accent": "#ed8936",       # Orange
    "success": "#38a169",      # Green
    "warning": "#d69e2e",      # Yellow
    "danger": "#e53e3e",       # Red
    "light": "#edf2f7",        # Light Gray
    "dark": "#1a202c",         # Near Black
}

CATEGORY_COLORS = {
    "Electronics": "#2c5282",
    "Clothing": "#ed8936",
    "Groceries": "#38a169",
    "Home & Kitchen": "#805ad5",
    "Beauty": "#d53f8c",
    "Books": "#d69e2e",
}

CITY_COLORS = {
    "Kathmandu": "#1a365d",
    "Pokhara": "#2c5282",
    "Lalitpur": "#3182ce",
    "Bhaktapur": "#4299e1",
    "Biratnagar": "#63b3ed",
    "Chitwan": "#90cdf4",
}

PAYMENT_COLORS = {
    "eSewa": "#38a169",
    "Khalti": "#2c5282",
    "Cash on Delivery": "#ed8936",
    "Bank Transfer": "#805ad5",
}


def setup_professional_style():
    """Configure matplotlib and seaborn for professional-looking charts."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 15,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    sns.set_theme(style="whitegrid", palette="muted")


def add_bar_labels(ax, fmt="{:,.0f}", fontsize=9):
    """Add value labels to bar containers."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=3)


def format_currency(value, decimals=0):
    """Format number as NPR currency."""
    if decimals == 0:
        return f"NPR {value:,.0f}"
    return f"NPR {value:,.{decimals}f}"


def format_number(value):
    """Format large numbers with K/M suffixes."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


def create_kpi_card(ax, value, label, color="#1a365d"):
    """Create a single KPI card for dashboard display."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="none",
        alpha=0.95
    ))
    
    ax.text(0.5, 0.65, value, fontsize=24, fontweight="bold",
            ha="center", va="center", color="white")
    ax.text(0.5, 0.35, label, fontsize=11,
            ha="center", va="center", color="white", alpha=0.9)


def create_kpi_dashboard(kpis, figsize=(16, 3)):
    """
    Create a KPI dashboard with multiple metric cards.
    
    Parameters:
        kpis: dict of {label: (value, color)}
    """
    n = len(kpis)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, (label, (value, color)) in zip(axes, kpis.items()):
        create_kpi_card(ax, value, label, color)
    
    plt.tight_layout()
    return fig, axes
