#!/usr/bin/env python3
"""
Deep Data Analysis for IITM Discourse Forum
Analyzes 500 users to extract engagement patterns, network dynamics, and outliers
"""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any

# Load data
with open('combined.json', 'r') as f:
    data = json.load(f)

users = data['directory_items']
print(f"🔍 Analyzing {len(users)} users from IITM Discourse Forum\n")

# ============================================================
# 1. ENGAGEMENT SEGMENTATION
# ============================================================
print("=" * 60)
print("📊 PHASE 1: ENGAGEMENT SEGMENTATION")
print("=" * 60)

# Calculate engagement score (composite metric)
def calculate_engagement_score(user):
    """Calculate a composite engagement score"""
    post_weight = 3
    topic_weight = 5
    solution_weight = 10
    like_given_weight = 1
    like_received_weight = 2
    
    score = (
        user['post_count'] * post_weight +
        user['topic_count'] * topic_weight +
        user['solutions'] * solution_weight +
        user['likes_given'] * like_given_weight +
        user['likes_received'] * like_received_weight
    )
    return score

# Add engagement scores
for user in users:
    user['engagement_score'] = calculate_engagement_score(user)

# Sort by engagement score
users_sorted = sorted(users, key=lambda x: x['engagement_score'], reverse=True)

# Define segments based on percentiles
engagement_scores = [u['engagement_score'] for u in users]
p90 = statistics.quantiles(engagement_scores, n=10)[8]  # 90th percentile
p75 = statistics.quantiles(engagement_scores, n=4)[2]   # 75th percentile
p50 = statistics.median(engagement_scores)              # 50th percentile
p25 = statistics.quantiles(engagement_scores, n=4)[0]   # 25th percentile

segments = {
    'Power Users': [],      # Top 10%
    'Active Contributors': [],  # 75-90%
    'Regular Members': [],  # 50-75%
    'Occasional Users': [], # 25-50%
    'Lurkers': []           # Bottom 25%
}

for user in users:
    score = user['engagement_score']
    if score >= p90:
        segments['Power Users'].append(user)
        user['segment'] = 'Power Users'
    elif score >= p75:
        segments['Active Contributors'].append(user)
        user['segment'] = 'Active Contributors'
    elif score >= p50:
        segments['Regular Members'].append(user)
        user['segment'] = 'Regular Members'
    elif score >= p25:
        segments['Occasional Users'].append(user)
        user['segment'] = 'Occasional Users'
    else:
        segments['Lurkers'].append(user)
        user['segment'] = 'Lurkers'

print("\n📈 User Segments Distribution:")
segment_counts = {}
for segment, seg_users in segments.items():
    count = len(seg_users)
    segment_counts[segment] = count
    print(f"  • {segment}: {count} users ({count/len(users)*100:.1f}%)")

# ============================================================
# 2. GROUP ANALYSIS (instead of temporal - no timestamps available)
# ============================================================
print("\n" + "=" * 60)
print("👥 PHASE 2: COMMUNITY GROUP ANALYSIS")
print("=" * 60)

# Analyze by primary group
group_stats = defaultdict(lambda: {
    'count': 0,
    'total_posts': 0,
    'total_likes_given': 0,
    'total_likes_received': 0,
    'total_solutions': 0,
    'total_days_visited': 0,
    'users': []
})

for user in users:
    group = user['user'].get('primary_group_name', 'No Group')
    if not group:
        group = 'No Group'
    
    stats = group_stats[group]
    stats['count'] += 1
    stats['total_posts'] += user['post_count']
    stats['total_likes_given'] += user['likes_given']
    stats['total_likes_received'] += user['likes_received']
    stats['total_solutions'] += user['solutions']
    stats['total_days_visited'] += user['days_visited']
    stats['users'].append(user['user']['name'] or user['user']['username'])

print("\n📊 Community Groups:")
group_analysis = []
for group, stats in sorted(group_stats.items(), key=lambda x: x[1]['count'], reverse=True):
    avg_posts = stats['total_posts'] / stats['count']
    avg_solutions = stats['total_solutions'] / stats['count']
    avg_days = stats['total_days_visited'] / stats['count']
    
    group_analysis.append({
        'group': group,
        'count': stats['count'],
        'avg_posts': round(avg_posts, 1),
        'avg_solutions': round(avg_solutions, 1),
        'avg_days_visited': round(avg_days, 1),
        'total_likes_received': stats['total_likes_received']
    })
    
    print(f"\n  [{group}] - {stats['count']} users")
    print(f"    Avg Posts: {avg_posts:.1f} | Avg Solutions: {avg_solutions:.1f} | Avg Days Visited: {avg_days:.1f}")

# Trust Level Analysis
print("\n\n📊 Trust Level Distribution:")
trust_levels = Counter(u['user']['trust_level'] for u in users)
trust_analysis = []
trust_labels = {0: 'New User', 1: 'Basic', 2: 'Member', 3: 'Regular', 4: 'Leader'}
for level, count in sorted(trust_levels.items()):
    label = trust_labels.get(level, f'Level {level}')
    trust_analysis.append({'level': level, 'label': label, 'count': count})
    print(f"  Trust Level {level} ({label}): {count} users")

# ============================================================
# 3. NETWORK DYNAMICS & HELPFULNESS
# ============================================================
print("\n" + "=" * 60)
print("🤝 PHASE 3: NETWORK DYNAMICS & HELPFULNESS")
print("=" * 60)

# Calculate helpfulness score
def calculate_helpfulness(user):
    """Helpfulness = (Likes Received + Solutions*5) / max(1, Posts Made)"""
    likes = user['likes_received']
    solutions = user['solutions']
    posts = max(1, user['post_count'])
    return (likes + solutions * 5) / posts

for user in users:
    user['helpfulness_score'] = calculate_helpfulness(user)
    user['generosity_ratio'] = user['likes_given'] / max(1, user['likes_received'])
    user['reading_ratio'] = user['posts_read'] / max(1, user['post_count'])  # How much they read vs write

# Top Helpers (high helpfulness score with significant activity)
active_users = [u for u in users if u['post_count'] >= 50]  # At least 50 posts
top_helpers = sorted(active_users, key=lambda x: x['helpfulness_score'], reverse=True)[:10]

print("\n🏆 Top 10 Most Helpful Users (min 50 posts):")
top_helpers_data = []
for i, user in enumerate(top_helpers, 1):
    name = user['user']['name'] or user['user']['username']
    top_helpers_data.append({
        'rank': i,
        'name': name,
        'username': user['user']['username'],
        'helpfulness_score': round(user['helpfulness_score'], 2),
        'solutions': user['solutions'],
        'likes_received': user['likes_received'],
        'posts': user['post_count']
    })
    print(f"  {i}. {name[:30]:<30} | Score: {user['helpfulness_score']:.2f} | Solutions: {user['solutions']} | Posts: {user['post_count']}")

# Most Generous (give more than receive)
most_generous = sorted(users, key=lambda x: x['generosity_ratio'], reverse=True)[:10]
print("\n💝 Top 10 Most Generous Users (likes given/received ratio):")
generous_data = []
for i, user in enumerate(most_generous[:5], 1):
    name = user['user']['name'] or user['user']['username']
    generous_data.append({
        'rank': i,
        'name': name,
        'username': user['user']['username'],
        'ratio': round(user['generosity_ratio'], 2),
        'given': user['likes_given'],
        'received': user['likes_received']
    })
    print(f"  {i}. {name[:30]:<30} | Ratio: {user['generosity_ratio']:.2f} | Given: {user['likes_given']} | Received: {user['likes_received']}")

# ============================================================
# 4. OUTLIER DETECTION - Hidden Gems
# ============================================================
print("\n" + "=" * 60)
print("💎 PHASE 4: OUTLIER DETECTION - HIDDEN GEMS")
print("=" * 60)

outliers = {}

# High Impact, Low Volume (high likes per post)
likes_per_post = [(u, u['likes_received'] / max(1, u['post_count'])) for u in users if u['post_count'] >= 10]
likes_per_post.sort(key=lambda x: x[1], reverse=True)
outliers['high_impact_low_volume'] = []
print("\n✨ High Impact, Low Volume (most likes per post, min 10 posts):")
for i, (user, ratio) in enumerate(likes_per_post[:5], 1):
    name = user['user']['name'] or user['user']['username']
    outliers['high_impact_low_volume'].append({
        'name': name,
        'username': user['user']['username'],
        'likes_per_post': round(ratio, 2),
        'total_likes': user['likes_received'],
        'posts': user['post_count']
    })
    print(f"  {i}. {name[:30]:<30} | {ratio:.2f} likes/post | {user['likes_received']} total likes | {user['post_count']} posts")

# Silent Readers (high posts_read, low post_count)
silent_readers = [(u, u['posts_read'] / max(1, u['post_count'])) for u in users if u['posts_read'] >= 1000]
silent_readers.sort(key=lambda x: x[1], reverse=True)
outliers['silent_readers'] = []
print("\n📚 Silent Readers (highest read-to-write ratio, min 1000 posts read):")
for i, (user, ratio) in enumerate(silent_readers[:5], 1):
    name = user['user']['name'] or user['user']['username']
    outliers['silent_readers'].append({
        'name': name,
        'username': user['user']['username'],
        'read_ratio': round(ratio, 1),
        'posts_read': user['posts_read'],
        'posts': user['post_count']
    })
    print(f"  {i}. {name[:30]:<30} | Reads {ratio:.0f}x more | {user['posts_read']} posts read | {user['post_count']} posts written")

# Marathon Users (highest days_visited)
marathon_users = sorted(users, key=lambda x: x['days_visited'], reverse=True)[:5]
outliers['marathon_users'] = []
print("\n🏃 Marathon Users (most consistent visitors):")
for i, user in enumerate(marathon_users, 1):
    name = user['user']['name'] or user['user']['username']
    outliers['marathon_users'].append({
        'name': name,
        'username': user['user']['username'],
        'days_visited': user['days_visited'],
        'posts': user['post_count'],
        'solutions': user['solutions']
    })
    print(f"  {i}. {name[:30]:<30} | {user['days_visited']} days visited | {user['post_count']} posts")

# Solution Masters (highest solutions)
solution_masters = sorted(users, key=lambda x: x['solutions'], reverse=True)[:5]
outliers['solution_masters'] = []
print("\n🎯 Solution Masters (most problems solved):")
for i, user in enumerate(solution_masters, 1):
    name = user['user']['name'] or user['user']['username']
    outliers['solution_masters'].append({
        'name': name,
        'username': user['user']['username'],
        'solutions': user['solutions'],
        'posts': user['post_count'],
        'efficiency': round(user['solutions'] / max(1, user['post_count']) * 100, 1)
    })
    print(f"  {i}. {name[:30]:<30} | {user['solutions']} solutions | {user['solutions']/max(1,user['post_count'])*100:.1f}% solution rate")

# Gamification Champions
gamification_champs = sorted(users, key=lambda x: x['gamification_score'], reverse=True)[:5]
outliers['gamification_champions'] = []
print("\n🎮 Gamification Champions (highest gamification score):")
for i, user in enumerate(gamification_champs, 1):
    name = user['user']['name'] or user['user']['username']
    outliers['gamification_champions'].append({
        'name': name,
        'username': user['user']['username'],
        'gamification_score': user['gamification_score'],
        'posts': user['post_count'],
        'days_visited': user['days_visited']
    })
    print(f"  {i}. {name[:30]:<30} | Score: {user['gamification_score']} | {user['days_visited']} days visited")

# ============================================================
# 5. STATISTICAL OVERVIEW
# ============================================================
print("\n" + "=" * 60)
print("📈 PHASE 5: STATISTICAL OVERVIEW")
print("=" * 60)

# Calculate key statistics
stats_overview = {
    'total_users': len(users),
    'total_posts': sum(u['post_count'] for u in users),
    'total_topics': sum(u['topic_count'] for u in users),
    'total_solutions': sum(u['solutions'] for u in users),
    'total_likes_given': sum(u['likes_given'] for u in users),
    'total_likes_received': sum(u['likes_received'] for u in users),
    'avg_posts_per_user': round(sum(u['post_count'] for u in users) / len(users), 1),
    'avg_days_visited': round(sum(u['days_visited'] for u in users) / len(users), 1),
    'avg_solutions_per_user': round(sum(u['solutions'] for u in users) / len(users), 1),
    'median_posts': statistics.median([u['post_count'] for u in users]),
    'median_likes_received': statistics.median([u['likes_received'] for u in users]),
    'avg_time_read_hours': round(sum(u['time_read'] for u in users) / len(users) / 3600, 1),
}

print(f"""
📊 Community Overview:
  • Total Users Analyzed: {stats_overview['total_users']}
  • Total Posts Created: {stats_overview['total_posts']:,}
  • Total Topics Started: {stats_overview['total_topics']:,}
  • Total Solutions Given: {stats_overview['total_solutions']:,}
  • Total Likes Exchanged: {stats_overview['total_likes_given'] + stats_overview['total_likes_received']:,}

📐 Per-User Averages:
  • Avg Posts/User: {stats_overview['avg_posts_per_user']}
  • Avg Days Visited/User: {stats_overview['avg_days_visited']}
  • Avg Solutions/User: {stats_overview['avg_solutions_per_user']}
  • Avg Time Reading/User: {stats_overview['avg_time_read_hours']} hours

📉 Medians (showing typical user):
  • Median Posts: {stats_overview['median_posts']}
  • Median Likes Received: {stats_overview['median_likes_received']}
""")

# ============================================================
# 6. DISTRIBUTION DATA FOR CHARTS
# ============================================================
print("\n" + "=" * 60)
print("📊 PREPARING CHART DATA")
print("=" * 60)

# Post count distribution (binned)
post_bins = [0, 10, 50, 100, 250, 500, 1000, float('inf')]
post_distribution = []
for i in range(len(post_bins) - 1):
    count = len([u for u in users if post_bins[i] <= u['post_count'] < post_bins[i+1]])
    label = f"{post_bins[i]}-{post_bins[i+1]-1 if post_bins[i+1] != float('inf') else '+'}"
    post_distribution.append({'range': label, 'count': count})

# Days visited distribution
days_bins = [0, 30, 100, 250, 500, 750, 1000, float('inf')]
days_distribution = []
for i in range(len(days_bins) - 1):
    count = len([u for u in users if days_bins[i] <= u['days_visited'] < days_bins[i+1]])
    label = f"{days_bins[i]}-{days_bins[i+1]-1 if days_bins[i+1] != float('inf') else '+'}"
    days_distribution.append({'range': label, 'count': count})

# Likes received distribution
likes_bins = [0, 50, 100, 200, 400, 600, 800, float('inf')]
likes_distribution = []
for i in range(len(likes_bins) - 1):
    count = len([u for u in users if likes_bins[i] <= u['likes_received'] < likes_bins[i+1]])
    label = f"{likes_bins[i]}-{likes_bins[i+1]-1 if likes_bins[i+1] != float('inf') else '+'}"
    likes_distribution.append({'range': label, 'count': count})

# Solutions distribution
solutions_bins = [0, 1, 5, 10, 25, 50, 100, float('inf')]
solutions_distribution = []
for i in range(len(solutions_bins) - 1):
    count = len([u for u in users if solutions_bins[i] <= u['solutions'] < solutions_bins[i+1]])
    label = f"{solutions_bins[i]}-{solutions_bins[i+1]-1 if solutions_bins[i+1] != float('inf') else '+'}"
    solutions_distribution.append({'range': label, 'count': count})

# Engagement score distribution by segment
engagement_by_segment = []
for segment, seg_users in segments.items():
    if seg_users:
        engagement_by_segment.append({
            'segment': segment,
            'count': len(seg_users),
            'avg_posts': round(sum(u['post_count'] for u in seg_users) / len(seg_users), 1),
            'avg_likes': round(sum(u['likes_received'] for u in seg_users) / len(seg_users), 1),
            'avg_solutions': round(sum(u['solutions'] for u in seg_users) / len(seg_users), 1),
            'avg_days': round(sum(u['days_visited'] for u in seg_users) / len(seg_users), 1)
        })

# ============================================================
# USER SPOTLIGHT DATA
# ============================================================
# Top 20 users for spotlight section
spotlight_users = []
for user in users_sorted[:20]:
    spotlight_users.append({
        'name': user['user']['name'] or user['user']['username'],
        'username': user['user']['username'],
        'title': user['user'].get('title') or '',
        'group': user['user'].get('primary_group_name') or 'No Group',
        'trust_level': user['user']['trust_level'],
        'posts': user['post_count'],
        'solutions': user['solutions'],
        'likes_received': user['likes_received'],
        'days_visited': user['days_visited'],
        'engagement_score': user['engagement_score'],
        'segment': user['segment'],
        'gamification_score': user['gamification_score']
    })

# ============================================================
# EXPORT ALL ANALYSIS TO JSON
# ============================================================
analysis_results = {
    'stats_overview': stats_overview,
    'segment_counts': segment_counts,
    'engagement_by_segment': engagement_by_segment,
    'group_analysis': group_analysis,
    'trust_analysis': trust_analysis,
    'top_helpers': top_helpers_data,
    'generous_users': generous_data,
    'outliers': outliers,
    'post_distribution': post_distribution,
    'days_distribution': days_distribution,
    'likes_distribution': likes_distribution,
    'solutions_distribution': solutions_distribution,
    'spotlight_users': spotlight_users
}

with open('analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print("\n✅ Analysis complete! Results saved to 'analysis_results.json'")
print("=" * 60)
