import streamlit as st
import pandas as pd
from utilsDash import (
    analyze_echo_chambers, analyze_polarization, analyze_algorithmic_bias,
    analyze_misinformation, analyze_network_structure,
    plot_sentiment_distribution, plot_engagement_by_category,
    plot_temporal_content_spread, plot_user_content_diversity,
    plot_category_distribution, plot_health_scores, plot_topic_polarization,
    plot_virality_by_category, compute_overall_health_score
)

st.set_page_config(page_title="Social Media Platform Analysis", layout="wide")
st.title("🧪 Social Media Platform Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your social_data.csv", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Convert numeric columns
        for col in ['sentiment', 'likes', 'comments', 'shares']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Run analyses
        echo_res = analyze_echo_chambers(df)
        pol_res = analyze_polarization(df)
        alg_bias = analyze_algorithmic_bias(df)
        misinfo = analyze_misinformation(df)
        net_res = analyze_network_structure(df)

        # Calculate bias scores
        bias_scores = [
            echo_res['diversity_stats']['mean'],
            1 - min(pol_res['polarization_score'] / 1.0, 1.0),
            min(alg_bias['bias_score'], 2.0) / 2.0,
            min(misinfo['amplification_ratio'], 3.0) / 3.0 if misinfo['amplification_ratio'] is not None else 0
        ]

        # Calculate overall health score
        overall = compute_overall_health_score({
            'echo_chambers': echo_res,
            'polarization': pol_res,
            'algorithmic_bias': alg_bias,
            'misinformation': misinfo,
        })

        st.header("📊 Summary Metrics")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Echo Chamber Strength", f"{echo_res['diversity_stats']['mean']:.3f}",
                              help="Higher values indicate more diverse content consumption (healthier)")
        metrics_cols[1].metric("Polarization Level", f"{pol_res['polarization_score']:.3f}",
                              help="Higher values indicate more polarized sentiment (less healthy)")
        metrics_cols[2].metric("Algorithmic Bias", f"{alg_bias['bias_score']:.3f}",
                              help="Values > 1 indicate harmful content gets more engagement")
        metrics_cols[3].metric("Misinformation Spread", f"{misinfo['amplification_ratio']:.3f}",
                              help="Higher values indicate misinformation spreads faster than safe content")

        st.subheader("🏥 Overall Platform Health Score")
        health_col1, health_col2 = st.columns([1, 3])
        with health_col1:
            st.info(f"**{overall:.1f}/100**")
        with health_col2:
            if overall > 80:
                st.success("✅ **EXCELLENT**: Platform shows healthy discourse patterns with minimal harmful content amplification")
            elif overall > 60:
                st.warning("⚠️ **MODERATE**: Some concerning patterns detected - monitor for potential issues")
            else:
                st.error("❌ **POOR**: Significant platform health issues detected - immediate intervention recommended")
        st.divider()

        # SECTION 1: POLARIZATION ANALYSIS
        st.header("📈 1. Polarization Analysis")
        st.markdown("""
        **What it measures**: How extreme and divided user opinions are on the platform.  
        - **High standard deviation**: polarized content  
        - **Bimodal distribution**: echo chambers  
        - **Normal distribution around 0**: balanced discourse
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            st.pyplot(plot_sentiment_distribution(df), use_container_width=False)
        with col2:
            st.subheader("Most Polarized Topics")
            st.pyplot(plot_topic_polarization(pol_res['topic_polarization']), use_container_width=False)

        st.markdown(f"""
        **Analysis Results**:
        - Polarization Score: **{pol_res['polarization_score']:.3f}**
        - Kurtosis: **{pol_res['kurtosis']:.3f}**
        - Most polarized hashtag: **{pol_res['topic_polarization'].index[0] if len(pol_res['topic_polarization']) > 0 else 'N/A'}**
        """)

        st.divider()

        # SECTION 2: ALGORITHMIC BIAS ANALYSIS
        st.header("⚖️ 2. Algorithmic Bias Analysis")
        st.markdown("""
        **What it measures**: Whether harmful content gets more reach than fair.
        - **Bias Score > 1.0**: harmful content gets more engagement.
        - **Higher virality in harmful categories**: algorithmic amplification.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Engagement by Category")
            st.pyplot(plot_engagement_by_category(df), use_container_width=False)
        with col2:
            st.subheader("Virality by Category")
            st.pyplot(plot_virality_by_category(alg_bias['virality_data']), use_container_width=False)
        st.markdown(f"""
        **Analysis Results**:
        - Algorithmic Bias Score: **{alg_bias['bias_score']:.3f}**
        - Virality Bias: **{alg_bias['virality_bias']:.3f}**
        """)

        st.divider()

        # SECTION 3: MISINFORMATION SPREAD ANALYSIS
        st.header("🚨 3. Misinformation Spread Analysis")
        st.markdown("""
        **What it measures**: Spread rate and reach of misinformation vs. legitimate content.
        - **Amplification Ratio > 1.0**: misinformation spreads faster.
        - **High user participation**: more users spreading misinformation.
        """)

        st.subheader("Temporal Content Spread")
        st.pyplot(plot_temporal_content_spread(df), use_container_width=False)
        st.markdown(f"""
        **Analysis Results**:
        - Amplification Ratio: **{misinfo['amplification_ratio']:.3f}**
        - Engagement Ratio: **{misinfo['engagement_ratio']:.3f}**
        - User Participation: **{misinfo['user_participation']:.1%}**
        """)

        st.divider()

        # SECTION 4: ECHO CHAMBERS ANALYSIS
        st.header("🔄 4. Echo Chambers Analysis")
        st.markdown("""
        **What it measures**: Diversity of user content consumption and network clustering.
        - **Higher diversity**: users see varied content.
        - **Low diversity**: users are in echo chambers.
        - **Network metrics**: how tightly communities are grouped.
        """)

        st.subheader("User Content Diversity")
        st.pyplot(plot_user_content_diversity(df), use_container_width=False)
        if net_res:
            net_col1, net_col2, net_col3 = st.columns(3)
            with net_col1:
                st.metric("Network Density", f"{net_res['density']:.3f}",
                          help="How connected users are (0-1, higher = more connected)")
            with net_col2:
                st.metric("Average Clustering", f"{net_res['avg_clustering']:.3f}",
                          help="How much users cluster (0-1, higher = more clustering)")
            with net_col3:
                st.metric("Average Degree", f"{net_res['avg_degree']:.1f}",
                          help="Avg connections per user")
        else:
            st.warning("⚠️ Network too small for meaningful analysis.")

        modularity_text = f"{echo_res['modularity']:.3f}" if echo_res['modularity'] is not None else 'N/A'
        st.markdown(f"""
        **Analysis Results**:
        - Mean Diversity Score: **{echo_res['diversity_stats']['mean']:.3f}**
        - Content Entropy: **{echo_res['content_entropy']:.3f}**
        - Network Modularity: **{modularity_text}**
        """)

        st.divider()

        # SECTION 5: CONTENT CATEGORIES OVERVIEW
        st.header("📊 5. Content Categories Overview")
        st.markdown("""
        **What it measures**: Proportion of Safe, Harmful, and Misinformation content.
        """)

        st.subheader("Category Distribution")
        st.pyplot(plot_category_distribution(df), use_container_width=False)
        category_stats = df['category'].value_counts(normalize=True) * 100
        st.markdown("**Category Breakdown:**")
        for category, percentage in category_stats.items():
            if category in ['Harmful', 'Misinformation']:
                st.markdown(f"- 🔴 **{category}**: {percentage:.1f}%")
            else:
                st.markdown(f"- 🟢 **{category}**: {percentage:.1f}%")

        st.divider()

        # SECTION 6: HEALTH SCORES
        st.header("🎯 6. Normalized Platform Health Scores")
        st.markdown("""
        **All metrics on a 0–1 scale for easy comparison.**
        - **Diversity: Higher = better**
        - **Polarization: Lower = better**
        - **Algorithmic Bias: Lower = better**
        - **Misinformation Spread: Lower = better**
        """)
        st.subheader("Comparative Health Metrics")
        st.pyplot(plot_health_scores(bias_scores), use_container_width=False)

        st.markdown("**Score Breakdown:**")
        score_labels = ['Diversity', 'Polarization Control', 'Bias Control', 'Misinformation Control']
        for label, score in zip(score_labels, bias_scores):
            if score > 0.8:
                st.markdown(f"- ✅ **{label}**: {score:.3f} (Excellent)")
            elif score > 0.6:
                st.markdown(f"- ⚠️ **{label}**: {score:.3f} (Moderate)")
            else:
                st.markdown(f"- ❌ **{label}**: {score:.3f} (Needs Attention)")

        st.divider()

        # RECOMMENDATIONS
        st.header("💡 Recommendations")
        recommendations = []
        if overall < 60:
            recommendations.append("🚨 **Critical**: Implement immediate content moderation measures")
        if pol_res['polarization_score'] > 0.8:
            recommendations.append("📊 **Polarization**: Promote diverse viewpoints and cross-cutting content")
        if alg_bias['bias_score'] > 1.5:
            recommendations.append("⚖️ **Algorithmic Bias**: Review recommendation algorithms for harmful content amplification")
        if misinfo['amplification_ratio'] > 2.0:
            recommendations.append("🚨 **Misinformation**: Strengthen fact-checking and misinformation detection systems")
        if echo_res['diversity_stats']['mean'] < 0.3:
            recommendations.append("🔄 **Echo Chambers**: Encourage users to engage with diverse content sources")
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.success("✅ Platform is performing well across all health metrics!")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.error("Please make sure your CSV contains the correct columns: user_id, timestamp, sentiment, likes, comments, shares, hashtags, category")
        st.markdown("""
        **Expected CSV format**:
        - `user_id`: Unique identifier for users
        - `timestamp`: Post timestamp (YYYY-MM-DD HH:MM:SS format)
        - `sentiment`: Sentiment score (-1 to 1)
        - `likes`: Number of likes (integer)
        - `comments`: Number of comments (integer)
        - `shares`: Number of shares (integer)
        - `hashtags`: Comma-separated hashtags
        - `category`: Content category (Safe/Harmful/Misinformation)
        """)
else:
    st.warning("📂 Please upload your social_data.csv file to get started.")

    with st.expander("📋 Expected Data Format", expanded=True):
        st.markdown("""
        **Required CSV columns:**
        - `user_id`: Unique user identifier  
        - `timestamp`: Post timestamp (YYYY-MM-DD HH:MM:SS)
        - `sentiment`: Sentiment score (-1.0 to 1.0)  
        - `likes, comments, shares`: Engagement metrics (integers)
        - `hashtags`: Comma-separated hashtags
        - `category`: Content type (Safe/Harmful/Misinformation)
        """)
        st.markdown("**Sample data format:**")
        sample_data = {
            'user_id': [1, 2, 3],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
            'sentiment': [0.5, -0.3, 0.8],
            'likes': [100, 50, 200],
            'comments': [20, 10, 30],
            'shares': [15, 5, 25],
            'hashtags': ['tech,ai', 'politics,news', 'health,wellness'],
            'category': ['Safe', 'Misinformation', 'Safe']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True, height=150)
