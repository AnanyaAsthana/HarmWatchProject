import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def analyze_echo_chambers(df):
    user_diversity = df.groupby('user_id')['hashtags'].nunique() / df.groupby('user_id')['hashtags'].count()
    diversity_stats = {
        'mean': user_diversity.mean(),
        'std': user_diversity.std(),
        'min': user_diversity.min(),
        'max': user_diversity.max(),
    }
    B = nx.Graph()
    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        tags = [tag.strip() for tag in str(row['hashtags']).split(',')]
        for tag in tags:
            B.add_edge(user, tag, weight=1)
    users = [n for n in B.nodes() if n.startswith('user_')]
    user_network = nx.Graph()
    for user1 in users:
        user1_tags = set(B[user1])
        for user2 in users:
            if user1 != user2:
                user2_tags = set(B[user2])
                if len(user1_tags & user2_tags):
                    user_network.add_edge(user1, user2)
    modularity = None
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity as nxmod
        comms = list(greedy_modularity_communities(user_network))
        if comms: modularity = nxmod(user_network, comms)
    except Exception:
        pass
    user_category_entropy = [
        -sum([p * np.log(p) for p in counts if p > 0])
        for _, user_df in df.groupby('user_id')
        for counts in [user_df['category'].value_counts(normalize=True)]
    ]
    avg_entropy = float(np.mean(user_category_entropy))
    return {
        'user_diversity': user_diversity,
        'diversity_stats': diversity_stats,
        'modularity': modularity,
        'content_entropy': avg_entropy,
    }

def analyze_polarization(df):
    sentiment = df['sentiment'].dropna().astype(float)
    polarization_score = float(np.std(sentiment))
    kurtosis = float((np.mean((sentiment-np.mean(sentiment))**4)/(np.var(sentiment)**2))-3)
    topic_pol = df.groupby('hashtags')['sentiment'].std().sort_values(ascending=False)
    return {
        'polarization_score': polarization_score,
        'kurtosis': kurtosis,
        'topic_polarization': topic_pol
    }

def analyze_algorithmic_bias(df):
    df = df.copy()
    df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
    df['virality'] = (df['shares'] + df['comments']) / (df['likes'] + 1)
    harmful_mask = df['category'].isin(['Harmful','Misinformation'])
    safe_mask = df['category']=='Safe'
    harmful_eng = df[harmful_mask]['total_engagement'].sum()
    safe_eng = df[safe_mask]['total_engagement'].sum()
    harmful_ratio = len(df[harmful_mask]) / len(df)
    engagement_ratio = (harmful_eng / (harmful_eng + safe_eng)) if (harmful_eng + safe_eng) else 0
    bias_score = (engagement_ratio / harmful_ratio) if harmful_ratio else 1
    harmful_vir = df[harmful_mask]['virality'].mean()
    safe_vir = df[safe_mask]['virality'].mean()
    virality_bias = (harmful_vir/safe_vir) if safe_vir else 1
    virality_data = df.groupby('category')['virality'].mean()
    return {
        'bias_score': float(bias_score),
        'virality_bias': float(virality_bias),
        'virality_data': virality_data,
    }

def analyze_misinformation(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
    misinfo = df[df['category']=='Misinformation']
    safe = df[df['category']=='Safe']
    misinfo_hourly = misinfo.resample('H', on='timestamp').size()
    safe_hourly = safe.resample('H', on='timestamp').size()
    amplification_ratio = float(misinfo_hourly.mean() / safe_hourly.mean()) if safe_hourly.mean() > 0 else 0
    misinfo_eng = float(misinfo['total_engagement'].mean() if not misinfo.empty else 0)
    safe_eng = float(safe['total_engagement'].mean() if not safe.empty else 1)
    engagement_ratio = float(misinfo_eng/safe_eng if safe_eng else 1)
    users_posting_misinfo = misinfo['user_id'].nunique()
    total_users = df['user_id'].nunique()
    user_participation = float(users_posting_misinfo/total_users) if total_users else None
    return {
        'amplification_ratio': amplification_ratio,
        'engagement_ratio': engagement_ratio,
        'user_participation': user_participation,
        'misinfo_hourly': misinfo_hourly,
        'safe_hourly': safe_hourly,
    }

def analyze_network_structure(df):
    B = nx.Graph()
    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        tags = [tag.strip() for tag in str(row['hashtags']).split(',')]
        for tag in tags: B.add_edge(user, tag, weight=1)
    users = [n for n in B.nodes() if n.startswith('user_')]
    user_network = nx.Graph()
    for user1 in users:
        user1_tags = set(B[user1])
        for user2 in users:
            if user1 != user2:
                user2_tags = set(B[user2])
                if len(user1_tags & user2_tags): user_network.add_edge(user1, user2)
    if len(user_network.nodes()):
        density = nx.density(user_network)
        avg_clustering = nx.average_clustering(user_network)
        avg_degree = float(np.mean([d for n, d in user_network.degree()]))
        return {
            'density': density, 'avg_clustering': avg_clustering, 'avg_degree': avg_degree,
        }
    else: return None

# -------- ALL PLOTS BELOW ARE SMALL, BALANCED, AND CLEAN --------

def plot_sentiment_distribution(df):
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    ax.hist(df['sentiment'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_title('Sentiment Distribution', fontsize=9)
    ax.set_xlabel('Sentiment Score', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.axvline(df['sentiment'].mean(), color='red', linestyle='--', label=f"Mean: {df['sentiment'].mean():.2f}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

def plot_engagement_by_category(df):
    df = df.copy()
    df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
    engagement_by_category = df.groupby('category')['total_engagement'].mean()
    colors = ['red' if cat in ['Harmful','Misinformation'] else 'green' for cat in engagement_by_category.index]
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    ax.bar(engagement_by_category.index, engagement_by_category.values, color=colors, alpha=0.7)
    ax.set_title('Engagement by Category', fontsize=9)
    ax.set_ylabel('Avg Engagement', fontsize=8)
    ax.tick_params(axis='x', labelsize=7, rotation=20)
    plt.tight_layout()
    return fig

def plot_temporal_content_spread(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    hourly_data = df.groupby([df['timestamp'].dt.hour, 'category']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(2.5, 1.6))  # small, balanced

    # Color mapping
    colors = {'Safe': '#4caf50', 'Harmful': '#f44336', 'Misinformation': '#ff9800'}
    for col in hourly_data.columns:
        ax.plot(hourly_data.index, hourly_data[col], label=col, color=colors.get(col, None), linewidth=1)
    ax.set_title('Content Spread by Hour', fontsize=8, pad=3)
    ax.set_xlabel('Hour', fontsize=7)
    ax.set_ylabel('Post Count', fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.75, borderpad=0.5)
    fig.tight_layout(pad=0.6)
    return fig

def plot_user_content_diversity(df):
    user_diversity = df.groupby('user_id')['hashtags'].nunique() / df.groupby('user_id')['hashtags'].count()
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    ax.hist(user_diversity, bins=20, alpha=0.7, edgecolor='black')
    ax.set_title('Content Diversity', fontsize=9)
    ax.set_xlabel('Diversity Score', fontsize=8)
    ax.set_ylabel('Users', fontsize=8)
    ax.axvline(user_diversity.mean(), color='red', linestyle='--', label=f'Mean: {user_diversity.mean():.2f}')
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

def plot_category_distribution(df):
    category_counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.pie(
        category_counts.values, 
        labels=category_counts.index, 
        autopct='%1.1f%%', 
        textprops={'fontsize': 7}
    )
    ax.set_title('Category Distribution', fontsize=8)
    plt.tight_layout()
    return fig

def plot_health_scores(bias_scores):
    fig, ax = plt.subplots(figsize=(2.4, 1.1))
    categories = ['Diversity', 'Polarization', 'Algorithmic Bias', 'Misinfo Spread']
    bars = ax.barh(categories, bias_scores, color=['#4682b4', '#ffa500', '#ee4444', '#8652ee'])
    ax.set_title('Platform Health (0–1)', fontsize=8, pad=3)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.03, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', ha='left', fontsize=8)
    plt.tight_layout(pad=0.4)
    return fig

def plot_topic_polarization(topic_polarization):
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    top_topics = topic_polarization.head(10)
    ax.barh(range(len(top_topics)), top_topics.values)
    ax.set_yticks(range(len(top_topics)))
    ax.set_yticklabels(top_topics.index, fontsize=7)
    ax.set_title('Top Polarized Topics', fontsize=9)
    ax.set_xlabel('StdDev', fontsize=8)
    plt.tight_layout()
    return fig

def plot_virality_by_category(virality_data):
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    ax.bar(virality_data.index, virality_data.values)
    ax.set_title('Virality by Category', fontsize=9)
    ax.set_ylabel('Virality', fontsize=8)
    ax.tick_params(axis='x', labelsize=7, rotation=20)
    plt.tight_layout()
    return fig

def compute_overall_health_score(results):
    echo_mean = results['echo_chambers']['diversity_stats']['mean']
    pol_score = results['polarization']['polarization_score']
    bias_score = results['algorithmic_bias']['bias_score']
    misinfo_ratio = results['misinformation']['amplification_ratio']

    return (
        (1 - min(pol_score / 1.0, 1.0)) * 0.25 +
        echo_mean * 0.25 +
        (1 - min((bias_score - 1) / 1.0, 1.0)) * 0.25 +
        (1 - min((misinfo_ratio - 1) / 2.0, 1.0)) * 0.25
    ) * 100
