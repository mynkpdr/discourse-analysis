#!/usr/bin/env python3
"""
IITM Discourse Community — Exploratory Data Analysis & Data Story Generator
============================================================================
Loads combined.json (500 Discourse users), performs comprehensive EDA,
generates visualizations, and produces a self-contained HTML data story.
"""

import json
import os
import base64
import io
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_FILE = "combined.json"
OUTPUT_DIR = "output"
HTML_FILE = "datastory.html"
CHART_DPI = 130
PALETTE = sns.color_palette("viridis", 8)
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.05)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """Load combined.json and flatten into a tidy DataFrame."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = raw["directory_items"]
    rows = []
    for item in items:
        user = item["user"]
        row = {
            "user_id": user["id"],
            "username": user["username"],
            "name": user.get("name", ""),
            "trust_level": user.get("trust_level", 0),
            "primary_group": user.get("primary_group_name", "unknown"),
            "title": user.get("title", ""),
            "is_admin": user.get("admin", False),
            # Activity metrics
            "solutions": item.get("solutions", 0),
            "gamification_score": item.get("gamification_score", 0),
            "likes_received": item.get("likes_received", 0),
            "likes_given": item.get("likes_given", 0),
            "topic_count": item.get("topic_count", 0),
            "post_count": item.get("post_count", 0),
            "topics_entered": item.get("topics_entered", 0),
            "posts_read": item.get("posts_read", 0),
            "days_visited": item.get("days_visited", 0),
            "time_read": item.get("time_read", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Clean nulls / fill defaults
    df["primary_group"] = df["primary_group"].fillna("unknown").replace("", "unknown")
    df["title"] = df["title"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["is_admin"] = df["is_admin"].fillna(False).astype(bool)

    # Derived metrics
    df["time_read_hours"] = (df["time_read"] / 3600).round(1)
    df["likes_ratio"] = (df["likes_given"] / df["likes_received"].replace(0, np.nan)).round(2)
    df["posts_per_day"] = (df["post_count"] / df["days_visited"].replace(0, np.nan)).round(3)
    df["reads_per_day"] = (df["posts_read"] / df["days_visited"].replace(0, np.nan)).round(1)
    df["solutions_per_post"] = (df["solutions"] / df["post_count"].replace(0, np.nan)).round(3)
    df["engagement_score"] = (
        df["likes_received"] + df["likes_given"] + df["solutions"] * 3 + df["post_count"]
    )

    # Role label
    def role_label(row):
        if row["is_admin"]:
            return "Admin"
        g = str(row["primary_group"]).lower()
        if "course_support" in g:
            return "Course Support"
        if "faculty" in g:
            return "Faculty"
        if "alumni" in g:
            return "Alumni"
        if "student" in g or "ds-" in g or "es-" in g:
            return "Student"
        return "Other"

    df["role"] = df.apply(role_label, axis=1)

    print(f"✓ Loaded {len(df)} users  |  Columns: {list(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute key summary statistics and return as a dict of DataFrames/values."""
    numeric_cols = [
        "solutions", "gamification_score", "likes_received", "likes_given",
        "topic_count", "post_count", "topics_entered", "posts_read",
        "days_visited", "time_read_hours", "engagement_score",
    ]
    summary = df[numeric_cols].describe().round(2)

    # Skewness
    skewness = df[numeric_cols].skew().round(2).rename("skewness")

    # Role breakdown
    role_stats = df.groupby("role")[numeric_cols].mean().round(1)

    # Trust level breakdown
    trust_stats = df.groupby("trust_level")[numeric_cols].mean().round(1)

    results = {
        "summary": summary,
        "skewness": skewness,
        "role_stats": role_stats,
        "trust_stats": trust_stats,
        "total_users": len(df),
        "total_posts": int(df["post_count"].sum()),
        "total_topics": int(df["topic_count"].sum()),
        "total_solutions": int(df["solutions"].sum()),
        "total_likes": int(df["likes_received"].sum()),
        "median_days_visited": int(df["days_visited"].median()),
        "median_time_hours": float(df["time_read_hours"].median()),
    }
    print("✓ Descriptive statistics computed")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOP / BOTTOM PERFORMERS
# ═══════════════════════════════════════════════════════════════════════════════

def top_bottom(df: pd.DataFrame, col: str, n: int = 10):
    """Return top-n and bottom-n (non-zero) users for a given metric."""
    top = df.nlargest(n, col)[["username", "name", "role", col]].reset_index(drop=True)
    nonzero = df[df[col] > 0]
    bottom = nonzero.nsmallest(n, col)[["username", "name", "role", col]].reset_index(drop=True)
    return top, bottom


def leaderboard_tables(df: pd.DataFrame) -> dict:
    """Build leaderboards for key metrics."""
    metrics = [
        ("likes_received", "Most Liked"),
        ("solutions", "Top Problem Solvers"),
        ("post_count", "Most Prolific Posters"),
        ("gamification_score", "Gamification Leaders"),
        ("days_visited", "Most Consistent Visitors"),
        ("time_read_hours", "Deepest Readers (Hours)"),
        ("engagement_score", "Overall Engagement"),
    ]
    boards = {}
    for col, label in metrics:
        top, _ = top_bottom(df, col, n=10)
        boards[label] = top
    print("✓ Leaderboards built")
    return boards


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATIONS & PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation matrix for activity metrics."""
    cols = [
        "likes_received", "likes_given", "post_count", "topic_count",
        "solutions", "gamification_score", "posts_read", "days_visited",
        "time_read_hours", "topics_entered",
    ]
    corr = df[cols].corr(method="spearman").round(2)
    print("✓ Correlation matrix computed (Spearman)")
    return corr


def behavioral_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Segment users into behavioral archetypes based on activity patterns."""
    def classify(row):
        if row["solutions"] >= 20 and row["post_count"] >= 100:
            return "🔧 Expert Helper"
        if row["likes_received"] >= 100 and row["post_count"] >= 50:
            return "⭐ Community Star"
        if row["posts_read"] >= 5000 and row["post_count"] < 20:
            return "👀 Silent Reader"
        if row["topic_count"] >= 30 and row["solutions"] < 5:
            return "❓ Question Asker"
        if row["days_visited"] >= 200 and row["post_count"] >= 20:
            return "🏠 Regular"
        if row["days_visited"] < 30:
            return "👋 Drive-by"
        return "📊 Casual"

    df = df.copy()
    df["segment"] = df.apply(classify, axis=1)
    seg_counts = df["segment"].value_counts()
    print(f"✓ Behavioral segments:\n{seg_counts.to_string()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def chart_distribution_grid(df: pd.DataFrame) -> str:
    """Grid of distributions for key metrics."""
    metrics = [
        ("likes_received", "Likes Received"),
        ("post_count", "Posts Written"),
        ("solutions", "Solutions Given"),
        ("days_visited", "Days Visited"),
        ("time_read_hours", "Reading Time (hrs)"),
        ("gamification_score", "Gamification Score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        data = df[col]
        ax.hist(data, bins=40, color=PALETTE[2], edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="#e74c3c", ls="--", lw=1.5, label=f"Median: {data.median():.0f}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Users")
        ax.legend(fontsize=9)
    fig.suptitle("Distribution of Key Metrics Across 500 Users", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_correlation_heatmap(corr: pd.DataFrame) -> str:
    """Heatmap of metric correlations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 9})
    ax.set_title("Spearman Correlation Between Engagement Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_role_comparison(df: pd.DataFrame) -> str:
    """Box plots comparing roles across metrics."""
    metrics = ["likes_received", "post_count", "solutions", "days_visited"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col in zip(axes, metrics):
        order = df.groupby("role")[col].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="role", y=col, ax=ax, order=order, palette="Set2",
                    showfliers=False)
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("How Different Roles Compare", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_top10_bars(df: pd.DataFrame) -> str:
    """Horizontal bar charts for top-10 across key metrics."""
    metrics = [
        ("likes_received", "Top 10 — Likes Received", "#3498db"),
        ("solutions", "Top 10 — Solutions", "#2ecc71"),
        ("post_count", "Top 10 — Posts Written", "#e67e22"),
        ("engagement_score", "Top 10 — Engagement Score", "#9b59b6"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title, color) in zip(axes.ravel(), metrics):
        top = df.nlargest(10, col).sort_values(col)
        ax.barh(top["username"], top[col], color=color, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=12)
        for i, v in enumerate(top[col]):
            ax.text(v + max(top[col]) * 0.01, i, f"{v:,.0f}", va="center", fontsize=9)
    fig.suptitle("Leaderboard Snapshots", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_segments_pie(df: pd.DataFrame) -> str:
    """Pie chart of behavioral segments."""
    seg = df["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        seg.values, labels=seg.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(seg)),
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("User Behavioral Segments", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_scatter_engagement(df: pd.DataFrame) -> str:
    """Scatter: posts_read vs likes_received colored by role."""
    fig, ax = plt.subplots(figsize=(10, 7))
    roles = df["role"].unique()
    colors = sns.color_palette("Set1", len(roles))
    for role, color in zip(roles, colors):
        subset = df[df["role"] == role]
        ax.scatter(subset["posts_read"], subset["likes_received"],
                   alpha=0.6, label=role, color=color, s=40, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Posts Read", fontsize=12)
    ax.set_ylabel("Likes Received", fontsize=12)
    ax.set_title("Reading Volume vs. Community Recognition", fontsize=14, fontweight="bold")
    ax.legend(title="Role", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_trust_level(df: pd.DataFrame) -> str:
    """Trust-level distribution and average engagement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Count
    tl = df["trust_level"].value_counts().sort_index()
    ax1.bar(tl.index.astype(str), tl.values, color=PALETTE[:len(tl)], edgecolor="white")
    ax1.set_title("Users by Trust Level", fontweight="bold")
    ax1.set_xlabel("Trust Level")
    ax1.set_ylabel("Count")
    for i, v in enumerate(tl.values):
        ax1.text(i, v + 3, str(v), ha="center", fontweight="bold")

    # Avg engagement by trust
    avg = df.groupby("trust_level")["engagement_score"].mean().sort_index()
    ax2.bar(avg.index.astype(str), avg.values, color=PALETTE[3:3 + len(avg)], edgecolor="white")
    ax2.set_title("Avg Engagement Score by Trust Level", fontweight="bold")
    ax2.set_xlabel("Trust Level")
    ax2.set_ylabel("Engagement Score")
    for i, v in enumerate(avg.values):
        ax2.text(i, v + 10, f"{v:.0f}", ha="center", fontweight="bold")

    fig.tight_layout()
    return fig_to_base64(fig)


def chart_pareto(df: pd.DataFrame) -> str:
    """Pareto analysis: what % of users generate what % of content."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, color, label in [
        ("post_count", "#3498db", "Posts"),
        ("likes_received", "#e74c3c", "Likes Received"),
        ("solutions", "#2ecc71", "Solutions"),
    ]:
        sorted_vals = df[col].sort_values(ascending=False).values
        cumulative = np.cumsum(sorted_vals) / sorted_vals.sum() * 100
        pct_users = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
        ax.plot(pct_users, cumulative, lw=2.5, label=label, color=color)

    ax.axhline(80, ls="--", color="gray", alpha=0.6)
    ax.axvline(20, ls="--", color="gray", alpha=0.6)
    ax.text(21, 82, "80/20 line", fontsize=10, color="gray")
    ax.set_xlabel("% of Users (ranked by contribution)", fontsize=12)
    ax.set_ylabel("% of Total Output (cumulative)", fontsize=12)
    ax.set_title("Pareto Analysis: How Concentrated Is Contribution?", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_givers_vs_receivers(df: pd.DataFrame) -> str:
    """Scatter: likes given vs likes received."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["likes_given"], df["likes_received"], alpha=0.5, s=35, c=PALETTE[4], edgecolor="white", lw=0.3)
    # diagonal
    lim = max(df["likes_given"].max(), df["likes_received"].max())
    ax.plot([0, lim], [0, lim], ls="--", color="gray", alpha=0.5, label="Equal give/receive")
    ax.set_xlabel("Likes Given", fontsize=12)
    ax.set_ylabel("Likes Received", fontsize=12)
    ax.set_title("Generosity vs. Recognition", fontsize=14, fontweight="bold")
    ax.legend()
    # Annotate outliers
    for _, row in df.nlargest(5, "likes_received").iterrows():
        ax.annotate(row["username"], (row["likes_given"], row["likes_received"]),
                    fontsize=8, alpha=0.8, ha="left",
                    xytext=(5, 5), textcoords="offset points")
    fig.tight_layout()
    return fig_to_base64(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NARRATIVE / INSIGHT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame, stats: dict, corr: pd.DataFrame) -> list:
    """Generate textual insights from the data."""
    insights = []

    # 1. Pareto insight
    posts_sorted = df["post_count"].sort_values(ascending=False).values
    cum = np.cumsum(posts_sorted) / posts_sorted.sum()
    pct_for_80 = (np.searchsorted(cum, 0.80) + 1) / len(cum) * 100
    insights.append({
        "title": "The Power Law of Participation",
        "text": (
            f"Just <strong>{pct_for_80:.0f}%</strong> of users produce <strong>80%</strong> of all posts. "
            f"The community follows a steep power-law distribution — a tiny core drives the conversation "
            f"while the vast majority are readers or occasional participants."
        ),
        "icon": "📊"
    })

    # 2. Silent majority
    silent = df[df["post_count"] < 10]
    readers = df[(df["posts_read"] > 1000) & (df["post_count"] < 10)]
    insights.append({
        "title": "The Silent Majority",
        "text": (
            f"<strong>{len(silent)}</strong> users ({len(silent)/len(df)*100:.0f}%) have written fewer than 10 posts, "
            f"yet <strong>{len(readers)}</strong> of them have read over 1,000 posts each. "
            f"These lurkers are deeply engaged consumers — they absorb knowledge without contributing visibly."
        ),
        "icon": "🤫"
    })

    # 3. Solutions heroes
    top_solver = df.nlargest(1, "solutions").iloc[0]
    total_sol = df["solutions"].sum()
    top5_sol = df.nlargest(5, "solutions")["solutions"].sum()
    insights.append({
        "title": "The Solution Heroes",
        "text": (
            f"<strong>{top_solver['username']}</strong> alone has provided <strong>{top_solver['solutions']}</strong> "
            f"solutions — that's {top_solver['solutions']/total_sol*100:.1f}% of all answers in the community. "
            f"The top 5 solvers account for <strong>{top5_sol/total_sol*100:.1f}%</strong> of all solutions. "
            f"Without them, many questions would go unanswered."
        ),
        "icon": "🦸"
    })

    # 4. Correlation insight
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_corr = corr_pairs[0]
    insights.append({
        "title": "The Strongest Link",
        "text": (
            f"The strongest correlation (Spearman ρ = <strong>{top_corr[2]:.2f}</strong>) is between "
            f"<strong>{top_corr[0].replace('_', ' ')}</strong> and <strong>{top_corr[1].replace('_', ' ')}</strong>. "
            f"Users who do one tend to do the other — these behaviors are deeply intertwined."
        ),
        "icon": "🔗"
    })

    # 5. Generosity gap
    givers = df.nlargest(10, "likes_given")
    receivers = df.nlargest(10, "likes_received")
    overlap = set(givers["username"]) & set(receivers["username"])
    insights.append({
        "title": "Generosity vs. Fame",
        "text": (
            f"Of the top 10 like-givers and top 10 like-receivers, only <strong>{len(overlap)}</strong> "
            f"appear in both lists. The most generous users are often different from the most recognized ones — "
            f"giving and receiving appreciation flow through different people."
        ),
        "icon": "💝"
    })

    # 6. Time investment
    heavy = df[df["time_read_hours"] > 100]
    avg_heavy_likes = heavy["likes_received"].mean()
    avg_all_likes = df["likes_received"].mean()
    insights.append({
        "title": "Does Time Pay Off?",
        "text": (
            f"<strong>{len(heavy)}</strong> users have spent over 100 hours reading the forum. "
            f"Their average likes received ({avg_heavy_likes:.0f}) is <strong>{avg_heavy_likes/avg_all_likes:.1f}×</strong> "
            f"the community average ({avg_all_likes:.0f}). Deep reading clearly correlates with earning recognition."
        ),
        "icon": "⏰"
    })

    # 7. Role dynamics
    role_avg = df.groupby("role")["engagement_score"].mean().sort_values(ascending=False)
    top_role = role_avg.index[0]
    insights.append({
        "title": "Who Drives Engagement?",
        "text": (
            f"<strong>{top_role}s</strong> have the highest average engagement score "
            f"(<strong>{role_avg.iloc[0]:.0f}</strong>), compared to the overall average of "
            f"<strong>{df['engagement_score'].mean():.0f}</strong>. "
            f"Their involvement is essential to keeping the community alive and responsive."
        ),
        "icon": "🎯"
    })

    # 8. Consistency
    consistent = df[df["days_visited"] >= 365]
    insights.append({
        "title": "The Year-Long Commitment",
        "text": (
            f"<strong>{len(consistent)}</strong> users have visited the forum on 365+ days — "
            f"roughly a full year of daily visits. These super-consistent members form "
            f"the reliable backbone of the community ({len(consistent)/len(df)*100:.1f}% of all users)."
        ),
        "icon": "📅"
    })

    print(f"✓ Generated {len(insights)} narrative insights")
    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HTML DATA STORY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_html(
    df: pd.DataFrame,
    stats: dict,
    boards: dict,
    charts: dict,
    insights: list,
    corr: pd.DataFrame,
) -> str:
    """Generate a self-contained, visually appealing HTML data story."""

    def img_tag(b64: str, alt: str = "") -> str:
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);">'

    def leaderboard_html(title: str, table: pd.DataFrame) -> str:
        rows_html = ""
        for i, (_, r) in enumerate(table.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}"
            cols = [c for c in table.columns if c not in ("username", "name", "role")]
            val = r[cols[0]] if cols else ""
            rows_html += f"""
            <tr>
                <td style="text-align:center;font-size:1.2em;">{medal}</td>
                <td><strong>{r['username']}</strong><br><small style="color:#888;">{r.get('name','')}</small></td>
                <td><span class="badge">{r.get('role','')}</span></td>
                <td style="text-align:right;font-weight:600;">{val:,.0f}</td>
            </tr>"""
        return f"""
        <div class="leaderboard-card">
            <h3>{title}</h3>
            <table class="lb-table">
                <thead><tr><th>#</th><th>User</th><th>Role</th><th>Value</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>"""

    # Segment summary
    seg_counts = df["segment"].value_counts()
    seg_html = ""
    for seg, count in seg_counts.items():
        pct = count / len(df) * 100
        seg_html += f'<div class="seg-item"><span class="seg-label">{seg}</span><span class="seg-count">{count} users ({pct:.1f}%)</span></div>'

    # Build leaderboard sections (show 4 key ones)
    lb_keys = list(boards.keys())[:4]
    lb_html = "".join(leaderboard_html(k, boards[k]) for k in lb_keys)

    # Insight cards
    insight_cards = ""
    for ins in insights:
        insight_cards += f"""
        <div class="insight-card">
            <div class="insight-icon">{ins['icon']}</div>
            <h3>{ins['title']}</h3>
            <p>{ins['text']}</p>
        </div>"""

    # Key metrics banner
    km = stats

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IITM Discourse Community — Data Story</title>
<style>
  :root {{
    --primary: #2c3e50;
    --accent: #3498db;
    --accent2: #2ecc71;
    --bg: #f8f9fa;
    --card: #ffffff;
    --text: #333;
    --muted: #7f8c8d;
    --radius: 14px;
    --shadow: 0 4px 24px rgba(0,0,0,0.07);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
  }}
  .hero {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    color: white;
    text-align: center;
    padding: 80px 20px 60px;
  }}
  .hero h1 {{
    font-size: 2.8em;
    font-weight: 800;
    margin-bottom: 12px;
    letter-spacing: -1px;
  }}
  .hero p {{
    font-size: 1.2em;
    opacity: 0.85;
    max-width: 640px;
    margin: 0 auto;
  }}
  .hero .subtitle {{
    font-size: 0.95em;
    opacity: 0.6;
    margin-top: 16px;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
  }}
  .section {{
    margin: 48px 0;
  }}
  .section-title {{
    font-size: 1.8em;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 8px;
    border-left: 5px solid var(--accent);
    padding-left: 16px;
  }}
  .section-desc {{
    color: var(--muted);
    margin-bottom: 28px;
    font-size: 1.05em;
  }}
  /* KPI Banner */
  .kpi-banner {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin: -40px auto 40px;
    max-width: 1100px;
    padding: 0 24px;
    position: relative;
    z-index: 10;
  }}
  .kpi-card {{
    background: var(--card);
    border-radius: var(--radius);
    padding: 24px 16px;
    text-align: center;
    box-shadow: var(--shadow);
  }}
  .kpi-value {{
    font-size: 2em;
    font-weight: 800;
    color: var(--accent);
  }}
  .kpi-label {{
    font-size: 0.85em;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  /* Cards */
  .card {{
    background: var(--card);
    border-radius: var(--radius);
    padding: 32px;
    box-shadow: var(--shadow);
    margin-bottom: 28px;
  }}
  .card h3 {{
    margin-bottom: 16px;
    color: var(--primary);
  }}
  /* Insight cards */
  .insights-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
  }}
  .insight-card {{
    background: var(--card);
    border-radius: var(--radius);
    padding: 28px;
    box-shadow: var(--shadow);
    border-top: 4px solid var(--accent);
    transition: transform 0.2s;
  }}
  .insight-card:hover {{ transform: translateY(-4px); }}
  .insight-icon {{ font-size: 2em; margin-bottom: 8px; }}
  .insight-card h3 {{ color: var(--primary); margin-bottom: 10px; font-size: 1.15em; }}
  .insight-card p {{ color: #555; font-size: 0.98em; }}
  /* Leaderboards */
  .lb-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
  }}
  .leaderboard-card {{
    background: var(--card);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
  }}
  .leaderboard-card h3 {{
    font-size: 1.05em;
    color: var(--primary);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #eee;
  }}
  .lb-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  .lb-table th {{ text-align: left; color: var(--muted); font-weight: 600; padding: 6px 4px; font-size: 0.85em; text-transform: uppercase; }}
  .lb-table td {{ padding: 7px 4px; border-top: 1px solid #f0f0f0; }}
  .badge {{
    display: inline-block;
    background: #e8f4fd;
    color: #2980b9;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.78em;
    font-weight: 600;
  }}
  /* Segments */
  .seg-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 16px;
  }}
  .seg-item {{
    background: var(--card);
    border-radius: 10px;
    padding: 14px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
    min-width: 160px;
  }}
  .seg-label {{ font-size: 1.1em; font-weight: 600; }}
  .seg-count {{ font-size: 0.85em; color: var(--muted); margin-top: 2px; }}
  /* Chart */
  .chart-container {{
    margin: 20px 0;
  }}
  /* Footer */
  .footer {{
    text-align: center;
    color: var(--muted);
    padding: 40px 20px;
    font-size: 0.9em;
    border-top: 1px solid #e0e0e0;
    margin-top: 60px;
  }}
  /* Narrative */
  .narrative {{
    font-size: 1.08em;
    color: #444;
    max-width: 780px;
    margin-bottom: 16px;
  }}
  .narrative strong {{ color: var(--primary); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 768px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .hero h1 {{ font-size: 2em; }}
    .kpi-banner {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<!-- ═══ HERO ═══ -->
<div class="hero">
  <h1>The IITM Discourse Community</h1>
  <p>An exploration of 500 of the most active members on the IIT Madras Online Degree Discourse forum — who they are, how they engage, and what makes this community tick.</p>
  <div class="subtitle">Data Story · February 2026</div>
</div>

<!-- ═══ KPI BANNER ═══ -->
<div class="kpi-banner">
  <div class="kpi-card"><div class="kpi-value">{km['total_users']}</div><div class="kpi-label">Users Analyzed</div></div>
  <div class="kpi-card"><div class="kpi-value">{km['total_posts']:,}</div><div class="kpi-label">Total Posts</div></div>
  <div class="kpi-card"><div class="kpi-value">{km['total_topics']:,}</div><div class="kpi-label">Topics Created</div></div>
  <div class="kpi-card"><div class="kpi-value">{km['total_solutions']:,}</div><div class="kpi-label">Solutions Given</div></div>
  <div class="kpi-card"><div class="kpi-value">{km['total_likes']:,}</div><div class="kpi-label">Likes Exchanged</div></div>
  <div class="kpi-card"><div class="kpi-value">{km['median_days_visited']}</div><div class="kpi-label">Median Days Visited</div></div>
</div>

<div class="container">

<!-- ═══ SECTION 1: THE BIG PICTURE ═══ -->
<div class="section">
  <h2 class="section-title">The Big Picture</h2>
  <p class="section-desc">What does activity look like across 500 of the most engaged users?</p>

  <div class="card">
    <p class="narrative">
      Most community metrics follow a <strong>heavily right-skewed</strong> distribution. A handful of power users
      dominate every metric — likes, posts, solutions — while the median user is far more modest.
      The median user has visited <strong>{km['median_days_visited']} days</strong> and spent
      <strong>{km['median_time_hours']:.0f} hours</strong> reading. Let's dig into the distributions:
    </p>
    <div class="chart-container">
      {img_tag(charts['distributions'], 'Distribution of key metrics')}
    </div>
  </div>
</div>

<!-- ═══ SECTION 2: KEY INSIGHTS ═══ -->
<div class="section">
  <h2 class="section-title">Key Insights</h2>
  <p class="section-desc">Eight data-backed findings that reveal the community's dynamics.</p>
  <div class="insights-grid">
    {insight_cards}
  </div>
</div>

<!-- ═══ SECTION 3: PARETO ═══ -->
<div class="section">
  <h2 class="section-title">The Pareto Principle in Action</h2>
  <p class="section-desc">How concentrated is community output?</p>
  <div class="card">
    <p class="narrative">
      The classic 80/20 rule applies — and it's even more extreme here. A small fraction of users
      generates the overwhelming majority of posts, likes, and solutions. This chart shows cumulative
      contribution curves:
    </p>
    <div class="chart-container">
      {img_tag(charts['pareto'], 'Pareto analysis')}
    </div>
  </div>
</div>

<!-- ═══ SECTION 4: LEADERBOARDS ═══ -->
<div class="section">
  <h2 class="section-title">Hall of Fame</h2>
  <p class="section-desc">The top performers across key dimensions.</p>
  <div class="lb-grid">
    {lb_html}
  </div>
</div>

<!-- ═══ SECTION 5: BEHAVIORAL SEGMENTS ═══ -->
<div class="section">
  <h2 class="section-title">Who Are These Users?</h2>
  <p class="section-desc">We classified every user into a behavioral archetype based on their activity patterns.</p>

  <div class="card">
    <div class="chart-container">
      {img_tag(charts['segments'], 'Behavioral segments')}
    </div>
    <div class="seg-grid">
      {seg_html}
    </div>
  </div>
</div>

<!-- ═══ SECTION 6: CORRELATIONS ═══ -->
<div class="section">
  <h2 class="section-title">What Drives What?</h2>
  <p class="section-desc">Spearman correlations reveal which behaviors go hand-in-hand.</p>
  <div class="card">
    <p class="narrative">
      Strong correlations emerge between reading and writing — users who read more posts tend to write more,
      receive more likes, and stay longer. The heatmap below maps out every metric pair:
    </p>
    <div class="chart-container">
      {img_tag(charts['correlation'], 'Correlation heatmap')}
    </div>
  </div>
</div>

<!-- ═══ SECTION 7: ENGAGEMENT DYNAMICS ═══ -->
<div class="section">
  <h2 class="section-title">Engagement Dynamics</h2>
  <p class="section-desc">Deeper views into reading, recognition, and generosity.</p>
  <div class="two-col">
    <div class="card">
      <h3>Reading Volume vs. Recognition</h3>
      <p class="narrative" style="font-size:0.95em;">Does reading more translate to being recognized? This scatter plot maps every user.</p>
      <div class="chart-container">
        {img_tag(charts['scatter_engagement'], 'Reading vs Likes')}
      </div>
    </div>
    <div class="card">
      <h3>Generosity vs. Recognition</h3>
      <p class="narrative" style="font-size:0.95em;">Are the biggest givers also the biggest receivers? The diagonal line represents parity.</p>
      <div class="chart-container">
        {img_tag(charts['givers_receivers'], 'Givers vs Receivers')}
      </div>
    </div>
  </div>
</div>

<!-- ═══ SECTION 8: ROLES & TRUST ═══ -->
<div class="section">
  <h2 class="section-title">Roles & Trust Levels</h2>
  <p class="section-desc">How do different community roles and trust levels shape participation?</p>
  <div class="two-col">
    <div class="card">
      <h3>Role Comparison</h3>
      <div class="chart-container">
        {img_tag(charts['role_comparison'], 'Role comparison')}
      </div>
    </div>
    <div class="card">
      <h3>Trust Level Analysis</h3>
      <div class="chart-container">
        {img_tag(charts['trust_level'], 'Trust levels')}
      </div>
    </div>
  </div>
</div>

<!-- ═══ SECTION 9: TOP PERFORMERS ═══ -->
<div class="section">
  <h2 class="section-title">Top 10 Leaderboards</h2>
  <p class="section-desc">The users who stand out across four critical metrics.</p>
  <div class="card">
    <div class="chart-container">
      {img_tag(charts['top10_bars'], 'Top 10 bar charts')}
    </div>
  </div>
</div>

</div><!-- /container -->

<div class="footer">
  <p>IITM Discourse Community Data Story · Generated from top-500 user directory data · February 2026</p>
  <p>Built with Python, pandas, matplotlib & seaborn</p>
</div>

</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IITM Discourse Community — EDA Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_and_clean(INPUT_FILE)

    # 2. Stats
    stats = descriptive_stats(df)

    # 3. Leaderboards
    boards = leaderboard_tables(df)

    # 4. Correlations
    corr = correlation_analysis(df)

    # 5. Segments
    df = behavioral_segments(df)

    # 6. Insights
    insights = generate_insights(df, stats, corr)

    # 7. Charts
    print("⏳ Generating charts...")
    charts = {
        "distributions": chart_distribution_grid(df),
        "correlation": chart_correlation_heatmap(corr),
        "role_comparison": chart_role_comparison(df),
        "top10_bars": chart_top10_bars(df),
        "segments": chart_segments_pie(df),
        "scatter_engagement": chart_scatter_engagement(df),
        "trust_level": chart_trust_level(df),
        "pareto": chart_pareto(df),
        "givers_receivers": chart_givers_vs_receivers(df),
    }
    print(f"✓ {len(charts)} charts generated")

    # 8. HTML
    html = build_html(df, stats, boards, charts, insights, corr)
    html_path = os.path.join(OUTPUT_DIR, HTML_FILE)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ HTML data story saved to {html_path}")

    # 9. Also save processed DataFrame
    csv_path = os.path.join(OUTPUT_DIR, "users_processed.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Processed CSV saved to {csv_path}")

    # Summary stats to console
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Users:           {stats['total_users']}")
    print(f"  Total Posts:     {stats['total_posts']:,}")
    print(f"  Total Topics:    {stats['total_topics']:,}")
    print(f"  Total Solutions: {stats['total_solutions']:,}")
    print(f"  Total Likes:     {stats['total_likes']:,}")
    print(f"  Median Days:     {stats['median_days_visited']}")
    print(f"  Median Read Hrs: {stats['median_time_hours']:.1f}")
    print("=" * 60)
    print("✅ Done! Open output/datastory.html in a browser.")


if __name__ == "__main__":
    main()
