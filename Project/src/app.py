"""
Streamlit application for Enhanced Community Detection visualization.
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
import os
import tempfile
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Set, Any, Optional, Union

# Import our modules
from enhanced_community_detection import EnhancedCommunityDetection
import data_utils
import analytics
import visualization

st.set_page_config(layout="wide", page_title="Enhanced Community Detection", page_icon="🔍")

# CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    font-weight: 700;
}
.sub-header {
    font-size: 1.5rem;
    color: #424242;
    font-weight: 600;
}
.section-header {
    font-size: 1.2rem;
    color: #1E88E5;
    font-weight: 600;
}
.info-text {
    font-size: 1rem;
    color: #616161;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">Enhanced Community Detection</p>', unsafe_allow_html=True)
st.markdown("""
This application demonstrates an enhanced community detection algorithm that improves upon the classic 
Greedy Modularity Optimization (GMO) by incorporating local clustering coefficients and network structural metrics.
""")

# Sidebar for navigation
st.sidebar.markdown("# Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Network Analysis", "Community Detection", "Comparison", "Advanced Features"]
)

# Session state to store our data and results
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'baseline_communities' not in st.session_state:
    st.session_state.baseline_communities = None
if 'enhanced_communities' not in st.session_state:
    st.session_state.enhanced_communities = None

# Function to generate network visualization with Plotly
def visualize_network_plotly(G, communities=None, community_map=None):
    # Use our modular visualization function
    return visualization.create_interactive_network(G, communities)


### HOME PAGE CONTENT ###
if page == "Home":
    st.markdown('<p class="sub-header">Upload or Generate a Network</p>', unsafe_allow_html=True)
    
    # Allow uploading a file or selecting a demo dataset
    data_source = st.radio(
        "Choose data source",
        ["Upload your own network", "Use a demo network", "Generate a synthetic network"]
    )
    
    if data_source == "Upload your own network":
        uploaded_file = st.file_uploader("Choose a network file", 
                                         type=["csv", "txt", "edgelist", "graphml", "gml", "gexf"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load the network
            is_directed = st.checkbox("Directed network")
            is_weighted = st.checkbox("Weighted network")
            
            try:
                G = data_utils.load_network(tmp_path, directed=is_directed, weighted=is_weighted)
                st.success(f"Network loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
                st.session_state.graph = G
                
                # Display a small preview of the network
                if G.number_of_nodes() <= 100:
                    fig = visualize_network_plotly(G)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Network is too large to display completely. Using a sample for visualization.")
                    sampled_G = data_utils.sample_network(G, 100, method='snowball')
                    fig = visualize_network_plotly(sampled_G)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading network: {str(e)}")
                os.unlink(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
    elif data_source == "Use a demo network":
        demo_option = st.selectbox(
            "Select a demo network",
            ["Zachary's Karate Club", "Les Misérables", "Political Blogs", "Facebook Social Circles"]
        )
        
        if demo_option == "Zachary's Karate Club":
            G = nx.karate_club_graph()
            st.session_state.graph = G
            st.success(f"Loaded Zachary's Karate Club network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            fig = visualize_network_plotly(G)
            st.plotly_chart(fig, use_container_width=True)
            
        elif demo_option == "Les Misérables":
            G = nx.les_miserables_graph()
            st.session_state.graph = G
            st.success(f"Loaded Les Misérables character co-occurrence network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            fig = visualize_network_plotly(G)
            st.plotly_chart(fig, use_container_width=True)
            
        elif demo_option == "Political Blogs":
            try:
                polblogs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "data", "external", "polblogs.gml")
                G = nx.read_gml(polblogs_path)
                G = G.to_undirected()  # Convert to undirected for community detection
                G = data_utils.largest_connected_component(G)
                st.session_state.graph = G
                st.success(f"Loaded Political Blogs network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
                
                # Only show a sample if it's too large
                if G.number_of_nodes() <= 100:
                    fig = visualize_network_plotly(G)
                else:
                    sampled_G = data_utils.sample_network(G, 100, method='snowball')
                    fig = visualize_network_plotly(sampled_G)
                    st.info("Showing a sample of 100 nodes from the network.")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading Political Blogs network: {str(e)}")
                
        elif demo_option == "Facebook Social Circles":
            try:
                # This is a placeholder - you might need to adjust this to a real dataset path
                st.info("This would load Facebook Social Circles data, but it's not included by default. Using a synthetic scale-free network instead.")
                G = nx.barabasi_albert_graph(100, 3)
                st.session_state.graph = G
                st.success(f"Generated a synthetic scale-free network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
                fig = visualize_network_plotly(G)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating network: {str(e)}")
    
    elif data_source == "Generate a synthetic network":
        col1, col2 = st.columns(2)
        
        with col1:
            network_type = st.selectbox(
                "Network model",
                ["Barabasi-Albert (Scale-free)", "Watts-Strogatz (Small-world)", 
                 "Erdos-Renyi (Random)", "Regular Graph"]
            )
            
        with col2:
            n_nodes = st.slider("Number of nodes", 10, 500, 100)
        
        additional_params = {}
        
        if network_type == "Barabasi-Albert (Scale-free)":
            m = st.slider("m (edges per new node)", 1, 10, 3)
            additional_params['m'] = m
            model_name = 'barabasi_albert'
            
        elif network_type == "Watts-Strogatz (Small-world)":
            k = st.slider("k (mean degree)", 2, 10, 4)
            p = st.slider("p (rewiring probability)", 0.0, 1.0, 0.1)
            additional_params['k'] = k
            additional_params['p'] = p
            model_name = 'watts_strogatz'
            
        elif network_type == "Erdos-Renyi (Random)":
            p = st.slider("p (edge probability)", 0.01, 1.0, 0.1)
            additional_params['p'] = p
            model_name = 'random'
            
        elif network_type == "Regular Graph":
            d = st.slider("d (degree of each node)", 2, min(10, n_nodes-1), 3)
            additional_params['d'] = d
            model_name = 'regular'
        
        if st.button("Generate Network"):
            try:
                G = data_utils.generate_synthetic_network(model_name, n_nodes, **additional_params)
                st.session_state.graph = G
                st.success(f"Generated a {network_type} network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
                fig = visualize_network_plotly(G)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating network: {str(e)}")
    
    # Instructions section            
    with st.expander("How to use this application"):
        st.markdown("""
        ### Getting Started
        1. **Upload your own network** or use a demo network
        2. Go to the **Network Analysis** page to examine network properties
        3. Run the **Community Detection** algorithms
        4. Compare the **baseline and enhanced community detection** results
        5. Explore **Advanced Features** like domain inference
        
        ### About the Algorithm
        Our enhanced community detection algorithm improves upon the classic Greedy Modularity Optimization by:
        
        1. Detecting initial communities using standard modularity optimization
        2. Identifying 'misfit' nodes with low clustering coefficient and weak internal connectivity
        3. Reassigning these nodes to better-fitting communities based on local structure analysis
        4. Reconstructing final communities with improved modularity and cohesion
        
        This approach often results in more coherent communities with higher modularity scores.
        """)

### NETWORK ANALYSIS PAGE ###
elif page == "Network Analysis":
    st.markdown('<p class="sub-header">Network Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.graph is None:
        st.warning("Please load a network on the Home page first.")
    else:
        G = st.session_state.graph
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        with col2:
            st.metric("Edges", G.number_of_edges())
        with col3:
            st.metric("Density", round(nx.density(G), 4))
            
        st.markdown('<p class="section-header">Network Properties</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate basic stats
            avg_clustering = nx.average_clustering(G)
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subG = G.subgraph(largest_cc).copy()
                avg_path_length = nx.average_shortest_path_length(subG)
                diameter = nx.diameter(subG)
                st.info("Network is not fully connected. Metrics calculated on largest connected component.")
            
            degrees = [d for _, d in G.degree()]
            avg_degree = sum(degrees) / len(degrees)
            
            metrics_df = pd.DataFrame({
                'Metric': ['Avg. Clustering Coefficient', 'Avg. Path Length', 'Diameter', 'Avg. Degree', 'Max Degree'],
                'Value': [
                    round(avg_clustering, 4),
                    round(avg_path_length, 4),
                    diameter,
                    round(avg_degree, 4),
                    max(degrees)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
        with col2:
            # Assortativity
            try:
                assortativity = nx.degree_assortativity_coefficient(G)
                st.metric("Degree Assortativity", round(assortativity, 4))
                
                if assortativity > 0.1:
                    st.info("Network shows assortative mixing (similar degree nodes tend to connect)")
                elif assortativity < -0.1:
                    st.info("Network shows disassortative mixing (different degree nodes tend to connect)")
                else:
                    st.info("Network shows neutral mixing patterns")
            except:
                st.warning("Could not calculate assortativity coefficient.")
        
        # Degree distribution
        st.markdown('<p class="section-header">Degree Distribution</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            degree_freq = nx.degree_histogram(G)
            degrees = range(len(degree_freq))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(degrees, degree_freq)
            ax.set_title("Degree Distribution")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
        with col2:
            # Log-log plot to check for power-law
            non_zero_degrees = [d for d in degrees if degree_freq[d] > 0]
            non_zero_freq = [degree_freq[d] for d in non_zero_degrees]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.loglog(non_zero_degrees, non_zero_freq, 'bo-')
            ax.set_title("Log-Log Degree Distribution")
            ax.set_xlabel("Degree (log)")
            ax.set_ylabel("Frequency (log)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        # Clustering coefficient distribution
        st.markdown('<p class="section-header">Clustering Coefficient Distribution</p>', unsafe_allow_html=True)
        
        clustering = nx.clustering(G)
        clustering_values = list(clustering.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(clustering_values, bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Local Clustering Coefficients")
        ax.set_xlabel("Clustering Coefficient")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        # Relationship between degree and clustering
        st.markdown('<p class="section-header">Degree vs Clustering</p>', unsafe_allow_html=True)
        
        degree_clustering_df = pd.DataFrame({
            'Node': list(G.nodes()),
            'Degree': [G.degree(n) for n in G.nodes()],
            'Clustering': [clustering[n] for n in G.nodes()]
        })
        
        fig = px.scatter(
            degree_clustering_df, x='Degree', y='Clustering',
            opacity=0.7,
            hover_data=['Node'],
            title='Relationship between Node Degree and Clustering Coefficient'
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

### COMMUNITY DETECTION PAGE ###
elif page == "Community Detection":
    st.markdown('<p class="sub-header">Community Detection</p>', unsafe_allow_html=True)
    
    if st.session_state.graph is None:
        st.warning("Please load a network on the Home page first.")
    else:
        G = st.session_state.graph
        
        st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            clustering_threshold = st.slider(
                "Clustering Coefficient Threshold",
                0.0, 1.0, 0.2,
                help="Nodes with clustering coefficient below this threshold will be considered for reassignment"
            )
            
        with col2:
            internal_connectivity_threshold = st.slider(
                "Internal Connectivity Threshold",
                0.0, 1.0, 0.3,
                help="Nodes with internal connectivity ratio below this threshold will be considered for reassignment"
            )
        
        # Run the algorithm
        if st.button("Detect Communities"):
            with st.spinner("Detecting communities... This may take a while for large networks."):
                try:
                    # Initialize the detector
                    detector = EnhancedCommunityDetection(G)
                    st.session_state.detector = detector
                    
                    # Detect baseline communities
                    baseline_communities = detector.detect_baseline_communities()
                    st.session_state.baseline_communities = baseline_communities
                    st.success(f"Detected {len(baseline_communities)} communities with baseline method.")
                    
                    # Enhance communities
                    enhanced_communities = detector.enhance_communities(
                        clustering_threshold=clustering_threshold,
                        internal_connectivity_threshold=internal_connectivity_threshold
                    )
                    st.session_state.enhanced_communities = enhanced_communities
                    st.success(f"Detected {len(enhanced_communities)} communities with enhanced method.")
                    
                    # Show modularity comparison
                    baseline_modularity = detector.calculate_modularity(baseline_communities)
                    enhanced_modularity = detector.calculate_modularity(enhanced_communities)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Baseline Modularity", round(baseline_modularity, 4))
                    
                    with col2:
                        diff = enhanced_modularity - baseline_modularity
                        st.metric("Enhanced Modularity", round(enhanced_modularity, 4), 
                                 round(diff, 4), delta_color="normal")
                    
                    # Visualize communities
                    st.markdown('<p class="section-header">Community Visualization</p>', unsafe_allow_html=True)
                    
                    # Use matplotlib for static visualization
                    fig, axes = detector.visualize_communities(method='both', figsize=(16, 8))
                    st.pyplot(fig)
                    
                    # Use plotly for interactive visualization
                    st.markdown('<p class="section-header">Interactive Visualization</p>', unsafe_allow_html=True)
                    
                    visualization_type = st.radio("Select visualization", ["Baseline Communities", "Enhanced Communities"], horizontal=True)
                    
                    if visualization_type == "Baseline Communities":
                        fig = visualize_network_plotly(G, detector.baseline_communities, detector.node_to_community_map_baseline)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = visualize_network_plotly(G, detector.enhanced_communities, detector.node_to_community_map_enhanced)
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during community detection: {str(e)}")
                    st.exception(e)
        
        # If detector exists in session state, show visualization options
        elif st.session_state.detector is not None:
            detector = st.session_state.detector
            
            # Show modularity comparison
            baseline_modularity = detector.calculate_modularity(detector.baseline_communities)
            enhanced_modularity = detector.calculate_modularity(detector.enhanced_communities)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Baseline Modularity", round(baseline_modularity, 4))
            
            with col2:
                diff = enhanced_modularity - baseline_modularity
                st.metric("Enhanced Modularity", round(enhanced_modularity, 4), 
                         round(diff, 4), delta_color="normal")
            
            # Visualize communities
            st.markdown('<p class="section-header">Community Visualization</p>', unsafe_allow_html=True)
            
            # Use matplotlib for static visualization
            fig, axes = detector.visualize_communities(method='both', figsize=(16, 8))
            st.pyplot(fig)
            
            # Use plotly for interactive visualization
            st.markdown('<p class="section-header">Interactive Visualization</p>', unsafe_allow_html=True)
            
            visualization_type = st.radio("Select visualization", ["Baseline Communities", "Enhanced Communities"], horizontal=True)
            
            if visualization_type == "Baseline Communities":
                fig = visualize_network_plotly(G, detector.baseline_communities, detector.node_to_community_map_baseline)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = visualize_network_plotly(G, detector.enhanced_communities, detector.node_to_community_map_enhanced)
                st.plotly_chart(fig, use_container_width=True)

### COMPARISON PAGE ###
elif page == "Comparison":
    st.markdown('<p class="sub-header">Community Structure Comparison</p>', unsafe_allow_html=True)
    
    if st.session_state.detector is None or st.session_state.baseline_communities is None or st.session_state.enhanced_communities is None:
        st.warning("Please run community detection on the Community Detection page first.")
    else:
        detector = st.session_state.detector
        baseline_communities = st.session_state.baseline_communities
        enhanced_communities = st.session_state.enhanced_communities
        G = st.session_state.graph
        
        # Get metrics
        metrics_df = detector.calculate_community_metrics()
        
        # Display the metrics
        st.markdown('<p class="section-header">Community Structure Metrics</p>', unsafe_allow_html=True)
        st.table(metrics_df)
        
        # Community size distribution
        st.markdown('<p class="section-header">Community Size Distribution</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_sizes = [len(comm) for comm in baseline_communities]
            fig, ax = plt.subplots()
            ax.hist(baseline_sizes, bins=min(15, len(baseline_communities)), alpha=0.7)
            ax.set_title("Baseline Community Size Distribution")
            ax.set_xlabel("Community Size")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
        with col2:
            enhanced_sizes = [len(comm) for comm in enhanced_communities]
            fig, ax = plt.subplots()
            ax.hist(enhanced_sizes, bins=min(15, len(enhanced_communities)), alpha=0.7)
            ax.set_title("Enhanced Community Size Distribution")
            ax.set_xlabel("Community Size")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
        # Node reassignment analysis
        if 'node_to_community_map_baseline' in dir(detector) and 'node_to_community_map_enhanced' in dir(detector):
            st.markdown('<p class="section-header">Node Reassignment Analysis</p>', unsafe_allow_html=True)
            
            # Create a dataframe of reassigned nodes
            reassigned_nodes = []
            
            for node in G.nodes():
                if (node in detector.node_to_community_map_baseline and 
                    node in detector.node_to_community_map_enhanced and
                    detector.node_to_community_map_baseline[node] != detector.node_to_community_map_enhanced[node]):
                    
                    reassigned_nodes.append({
                        'Node': node,
                        'Original Community': detector.node_to_community_map_baseline[node],
                        'New Community': detector.node_to_community_map_enhanced[node],
                        'Degree': G.degree(node),
                        'Clustering': detector.clustering_coefficients[node]
                    })
            
            if reassigned_nodes:
                reassigned_df = pd.DataFrame(reassigned_nodes)
                
                st.write(f"{len(reassigned_nodes)} nodes were reassigned to different communities:")
                st.dataframe(reassigned_df, use_container_width=True)
                
                # Analyze characteristics of reassigned nodes
                avg_degree_all = np.mean([G.degree(n) for n in G.nodes()])
                avg_clustering_all = np.mean(list(detector.clustering_coefficients.values()))
                
                avg_degree_reassigned = np.mean(reassigned_df['Degree'])
                avg_clustering_reassigned = np.mean(reassigned_df['Clustering'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Avg Degree (All Nodes)", round(avg_degree_all, 2))
                    st.metric("Avg Degree (Reassigned)", round(avg_degree_reassigned, 2))
                
                with col2:
                    st.metric("Avg Clustering (All Nodes)", round(avg_clustering_all, 4))
                    st.metric("Avg Clustering (Reassigned)", round(avg_clustering_reassigned, 4))
                    
                # Visualize reassigned nodes
                st.markdown('<p class="section-header">Reassigned Nodes Visualization</p>', unsafe_allow_html=True)
                
                # Create a copy of the graph with node colors
                highlighted_G = G.copy()
                color_map = []
                
                reassigned_set = set([node['Node'] for node in reassigned_nodes])
                
                for node in highlighted_G.nodes():
                    if node in reassigned_set:
                        color_map.append('red')
                    else:
                        color_map.append('lightblue')
                
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(highlighted_G, seed=42)
                nx.draw_networkx_nodes(highlighted_G, pos, node_color=color_map, alpha=0.8, node_size=80)
                nx.draw_networkx_edges(highlighted_G, pos, alpha=0.2)
                ax.set_title("Network with Reassigned Nodes Highlighted")
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No nodes were reassigned between communities.")
        
        # Inter-community connectivity
        st.markdown('<p class="section-header">Inter-community Connectivity</p>', unsafe_allow_html=True)
        
        # For baseline communities
        baseline_inter_community_edges = 0
        total_baseline_edges = 0
        
        for u, v in G.edges():
            total_baseline_edges += 1
            if (u in detector.node_to_community_map_baseline and 
                v in detector.node_to_community_map_baseline and
                detector.node_to_community_map_baseline[u] != detector.node_to_community_map_baseline[v]):
                baseline_inter_community_edges += 1
        
        # For enhanced communities
        enhanced_inter_community_edges = 0
        total_enhanced_edges = 0
        
        for u, v in G.edges():
            total_enhanced_edges += 1
            if (u in detector.node_to_community_map_enhanced and 
                v in detector.node_to_community_map_enhanced and
                detector.node_to_community_map_enhanced[u] != detector.node_to_community_map_enhanced[v]):
                enhanced_inter_community_edges += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_ratio = baseline_inter_community_edges / total_baseline_edges if total_baseline_edges > 0 else 0
            st.metric("Inter-community Edge Ratio (Baseline)", f"{baseline_ratio:.2%}")
            
        with col2:
            enhanced_ratio = enhanced_inter_community_edges / total_enhanced_edges if total_enhanced_edges > 0 else 0
            st.metric("Inter-community Edge Ratio (Enhanced)", f"{enhanced_ratio:.2%}")
        
        st.info("A lower ratio of inter-community edges typically indicates better-separated communities.")

### ADVANCED FEATURES PAGE ###
elif page == "Advanced Features":
    st.markdown('<p class="sub-header">Advanced Features</p>', unsafe_allow_html=True)
    
    if st.session_state.detector is None or st.session_state.enhanced_communities is None:
        st.warning("Please run community detection on the Community Detection page first.")
    else:
        detector = st.session_state.detector
        G = st.session_state.graph
        
        st.markdown('<p class="section-header">Domain Inference from Network Structure</p>', unsafe_allow_html=True)
        
        st.markdown("""
        This feature attempts to infer potential 'domains' or 'topics' for nodes based on community structure.
        For example, in a scientific citation network, domains might correspond to fields like Computer Science, Biology, etc.
        """)
        
        num_domains = st.slider("Number of domains to infer", 2, 10, 2)
        
        if st.button("Infer Node Domains"):
            with st.spinner("Inferring domains from network structure..."):
                try:
                    # Run the domain inference
                    node_domains = detector.infer_node_domains(num_domains)
                    
                    # Count nodes per domain
                    domain_counts = Counter(node_domains.values())
                    
                    # Display domain sizes
                    st.markdown("### Domain Distribution")
                    
                    domain_df = pd.DataFrame({
                        'Domain': list(domain_counts.keys()),
                        'Number of Nodes': list(domain_counts.values()),
                        'Percentage': [count / len(node_domains) * 100 for count in domain_counts.values()]
                    })
                    
                    st.dataframe(domain_df, use_container_width=True)
                    
                    # Create a pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(domain_counts.values(), labels=domain_counts.keys(), autopct='%1.1f%%', shadow=True)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)
                    
                    # Show domain overlap with communities
                    st.markdown("### Domain-Community Overlap")
                    
                    # Create a matrix of domain-community overlap
                    overlap_data = {}
                    
                    for domain in set(node_domains.values()):
                        domain_nodes = set([node for node, d in node_domains.items() if d == domain])
                        community_overlaps = {}
                        
                        for i, community in enumerate(detector.enhanced_communities):
                            overlap = len(domain_nodes.intersection(community))
                            community_overlaps[f"Community {i+1}"] = overlap
                            
                        overlap_data[domain] = community_overlaps
                    
                    overlap_df = pd.DataFrame(overlap_data)
                    st.dataframe(overlap_df, use_container_width=True)
                    
                    # Visualize network with domains
                    st.markdown("### Network Visualization by Domain")
                    
                    # Create a mapping of domain names to integers
                    domain_to_int = {d: i for i, d in enumerate(set(node_domains.values()))}
                    
                    # Create a dictionary for visualization
                    domain_map = {node: domain_to_int[domain] for node, domain in node_domains.items()}
                    domain_comms = []
                    
                    for d in set(node_domains.values()):
                        domain_comms.append(set([node for node, domain in node_domains.items() if domain == d]))
                    
                    # Visualize using Plotly
                    fig = visualize_network_plotly(G, domain_comms, domain_map)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during domain inference: {str(e)}")
                    st.exception(e)
        
        # Community cohesion analysis
        st.markdown('<p class="section-header">Community Cohesion Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        This analysis evaluates the internal structure and cohesion of each detected community.
        It can help identify which communities are most stable and which might benefit from further refinement.
        """)
        
        if st.button("Analyze Community Cohesion"):
            with st.spinner("Analyzing community cohesion..."):
                try:
                    # Analyze each community
                    community_metrics = []
                    
                    for i, community in enumerate(detector.enhanced_communities):
                        if len(community) < 2:
                            continue
                            
                        # Extract the subgraph
                        subgraph = G.subgraph(community)
                        
                        # Calculate metrics
                        metrics = {
                            'Community': i+1,
                            'Size': len(community),
                            'Internal Density': nx.density(subgraph),
                            'Avg. Clustering': nx.average_clustering(subgraph),
                            'Diameter': nx.diameter(subgraph) if nx.is_connected(subgraph) else float('inf')
                        }
                        
                        # Calculate conductance
                        internal_edges = subgraph.number_of_edges()
                        
                        # Count edges leaving the community
                        external_edges = 0
                        for node in community:
                            for neighbor in G.neighbors(node):
                                if neighbor not in community:
                                    external_edges += 1
                        
                        if internal_edges + external_edges > 0:
                            metrics['Conductance'] = external_edges / (2 * internal_edges + external_edges)
                        else:
                            metrics['Conductance'] = 0
                            
                        community_metrics.append(metrics)
                    
                    # Create a DataFrame
                    cohesion_df = pd.DataFrame(community_metrics)
                    
                    # Display the metrics
                    st.dataframe(cohesion_df, use_container_width=True)
                    
                    # Visualize metrics
                    st.markdown("### Community Cohesion Metrics")
                    
                    fig = px.scatter(
                        cohesion_df, x='Size', y='Internal Density',
                        color='Conductance', size='Size',
                        hover_data=['Community', 'Avg. Clustering'],
                        title='Community Cohesion Analysis',
                        labels={'Internal Density': 'Internal Density', 'Size': 'Community Size'},
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add quality score
                    cohesion_df['Quality Score'] = (
                        cohesion_df['Internal Density'] * 0.4 + 
                        cohesion_df['Avg. Clustering'] * 0.4 - 
                        cohesion_df['Conductance'] * 0.2
                    )
                    
                    # Sort by quality score
                    sorted_df = cohesion_df.sort_values('Quality Score', ascending=False).reset_index(drop=True)
                    
                    st.markdown("### Communities Ranked by Quality")
                    st.dataframe(sorted_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during cohesion analysis: {str(e)}")
                    st.exception(e)

# Add footer with credits
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    Enhanced Modularity-Based Community Detection Project
    </div>
    """, 
    unsafe_allow_html=True
)
