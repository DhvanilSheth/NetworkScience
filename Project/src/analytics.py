"""
Analytics module for analyzing community detection results.

This module provides functions to analyze and compare community detection results,
generate performance metrics, and summarize algorithm improvements.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict


def calculate_community_metrics(G: nx.Graph, 
                               communities: List[Set[int]]) -> Dict[str, float]:
    """
    Calculate metrics for community quality.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    communities : List[Set[int]]
        Communities as a list of sets of node ids
        
    Returns:
    --------
    Dict[str, float] : Dictionary of community metrics
    """
    metrics = {}
    
    # Calculate modularity
    metrics['modularity'] = nx.community.modularity(G, communities)
    
    # Calculate number of communities
    metrics['num_communities'] = len(communities)
    
    # Calculate community sizes
    community_sizes = [len(c) for c in communities]
    metrics['avg_community_size'] = np.mean(community_sizes)
    metrics['min_community_size'] = np.min(community_sizes)
    metrics['max_community_size'] = np.max(community_sizes)
    metrics['std_community_size'] = np.std(community_sizes)
    
    # Calculate community density
    densities = []
    for community in communities:
        if len(community) > 1:  # Need at least 2 nodes to calculate density
            subgraph = G.subgraph(community)
            densities.append(nx.density(subgraph))
    
    if densities:
        metrics['avg_community_density'] = np.mean(densities)
        metrics['min_community_density'] = np.min(densities)
        metrics['max_community_density'] = np.max(densities)
    else:
        metrics['avg_community_density'] = 0
        metrics['min_community_density'] = 0
        metrics['max_community_density'] = 0
    
    return metrics


def compare_community_assignments(baseline: List[Set[int]], 
                                 enhanced: List[Set[int]]) -> Dict[str, Any]:
    """
    Compare two community assignments and identify differences.
    
    Parameters:
    -----------
    baseline : List[Set[int]]
        Baseline communities
    enhanced : List[Set[int]]
        Enhanced communities
        
    Returns:
    --------
    Dict[str, Any] : Comparison metrics
    """
    # Create node to community maps
    baseline_map = {}
    for comm_id, comm in enumerate(baseline):
        for node in comm:
            baseline_map[node] = comm_id
            
    enhanced_map = {}
    for comm_id, comm in enumerate(enhanced):
        for node in comm:
            enhanced_map[node] = comm_id
    
    # Find nodes that changed community
    changed_nodes = []
    for node in baseline_map:
        if node in enhanced_map and baseline_map[node] != enhanced_map[node]:
            changed_nodes.append(node)
    
    # Calculate metrics
    comparison = {
        'baseline_communities': len(baseline),
        'enhanced_communities': len(enhanced),
        'changed_nodes': len(changed_nodes),
        'changed_nodes_percentage': len(changed_nodes) / len(baseline_map) * 100,
        'changed_node_list': changed_nodes
    }
    
    return comparison


def analyze_community_structure(G: nx.Graph, 
                              communities: List[Set[int]]) -> Dict[str, Any]:
    """
    Analyze internal structure of communities.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    communities : List[Set[int]]
        Communities as a list of sets of node ids
        
    Returns:
    --------
    Dict[str, Any] : Structure metrics
    """
    # Initialize metrics
    metrics = {
        'internal_edge_ratio': [],
        'external_edge_ratio': [],
        'conductance': [],
        'avg_clustering': []
    }
    
    # Create a mapping of nodes to their community ID
    node_to_community = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            node_to_community[node] = comm_id
    
    # Analyze each community
    for comm_id, community in enumerate(communities):
        # Skip single-node communities
        if len(community) <= 1:
            continue
            
        # Count internal and external edges
        internal_edges = 0
        external_edges = 0
        
        for node in community:
            for neighbor in G.neighbors(node):
                if neighbor in community:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Count each internal edge only once (undirected graph)
        internal_edges = internal_edges / 2
        
        # Calculate metrics
        total_edges = internal_edges + external_edges
        if total_edges > 0:
            metrics['internal_edge_ratio'].append(internal_edges / total_edges)
            metrics['external_edge_ratio'].append(external_edges / total_edges)
            
            # Conductance = external edges / total possible edges from community
            if external_edges > 0:
                metrics['conductance'].append(external_edges / (2 * internal_edges + external_edges))
            else:
                metrics['conductance'].append(0)
        
        # Calculate average clustering within community
        subgraph = G.subgraph(community)
        avg_clustering = nx.average_clustering(subgraph)
        metrics['avg_clustering'].append(avg_clustering)
    
    # Calculate average metrics
    overall_metrics = {
        'avg_internal_edge_ratio': np.mean(metrics['internal_edge_ratio']) if metrics['internal_edge_ratio'] else 0,
        'avg_external_edge_ratio': np.mean(metrics['external_edge_ratio']) if metrics['external_edge_ratio'] else 0,
        'avg_conductance': np.mean(metrics['conductance']) if metrics['conductance'] else 0,
        'avg_community_clustering': np.mean(metrics['avg_clustering']) if metrics['avg_clustering'] else 0
    }
    
    return overall_metrics


def generate_improvement_summary(G: nx.Graph,
                              baseline_communities: List[Set[int]],
                              enhanced_communities: List[Set[int]]) -> pd.DataFrame:
    """
    Generate a summary of improvements from baseline to enhanced communities.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    baseline_communities : List[Set[int]]
        Baseline communities
    enhanced_communities : List[Set[int]]
        Enhanced communities
        
    Returns:
    --------
    pd.DataFrame : Summary of improvements
    """
    # Calculate metrics
    baseline_metrics = calculate_community_metrics(G, baseline_communities)
    enhanced_metrics = calculate_community_metrics(G, enhanced_communities)
    
    baseline_structure = analyze_community_structure(G, baseline_communities)
    enhanced_structure = analyze_community_structure(G, enhanced_communities)
    
    comparison = compare_community_assignments(baseline_communities, enhanced_communities)
    
    # Create summary dataframe
    data = {
        'Metric': [],
        'Baseline': [],
        'Enhanced': [],
        'Change': [],
        'Improvement': []
    }
    
    # Add modularity
    data['Metric'].append('Modularity')
    data['Baseline'].append(baseline_metrics['modularity'])
    data['Enhanced'].append(enhanced_metrics['modularity'])
    change = enhanced_metrics['modularity'] - baseline_metrics['modularity']
    data['Change'].append(change)
    data['Improvement'].append('✓' if change > 0 else '✗')
    
    # Add number of communities
    data['Metric'].append('Number of Communities')
    data['Baseline'].append(baseline_metrics['num_communities'])
    data['Enhanced'].append(enhanced_metrics['num_communities'])
    change = enhanced_metrics['num_communities'] - baseline_metrics['num_communities']
    data['Change'].append(change)
    # More communities isn't necessarily better or worse
    data['Improvement'].append('-')
    
    # Add average community density
    data['Metric'].append('Avg Community Density')
    data['Baseline'].append(baseline_metrics['avg_community_density'])
    data['Enhanced'].append(enhanced_metrics['avg_community_density'])
    change = enhanced_metrics['avg_community_density'] - baseline_metrics['avg_community_density']
    data['Change'].append(change)
    data['Improvement'].append('✓' if change > 0 else '✗')
    
    # Add average internal edge ratio
    data['Metric'].append('Avg Internal Edge Ratio')
    data['Baseline'].append(baseline_structure['avg_internal_edge_ratio'])
    data['Enhanced'].append(enhanced_structure['avg_internal_edge_ratio'])
    change = enhanced_structure['avg_internal_edge_ratio'] - baseline_structure['avg_internal_edge_ratio']
    data['Change'].append(change)
    data['Improvement'].append('✓' if change > 0 else '✗')
    
    # Add average conductance (lower is better for conductance)
    data['Metric'].append('Avg Conductance')
    data['Baseline'].append(baseline_structure['avg_conductance'])
    data['Enhanced'].append(enhanced_structure['avg_conductance'])
    change = enhanced_structure['avg_conductance'] - baseline_structure['avg_conductance']
    data['Change'].append(change)
    data['Improvement'].append('✓' if change < 0 else '✗')
    
    # Add average community clustering
    data['Metric'].append('Avg Community Clustering')
    data['Baseline'].append(baseline_structure['avg_community_clustering'])
    data['Enhanced'].append(enhanced_structure['avg_community_clustering'])
    change = enhanced_structure['avg_community_clustering'] - baseline_structure['avg_community_clustering']
    data['Change'].append(change)
    data['Improvement'].append('✓' if change > 0 else '✗')
    
    return pd.DataFrame(data)


def plot_community_comparison(G: nx.Graph,
                            baseline_communities: List[Set[int]],
                            enhanced_communities: List[Set[int]],
                            figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Create a plot comparing community metrics before and after enhancement.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    baseline_communities : List[Set[int]]
        Baseline communities
    enhanced_communities : List[Set[int]]
        Enhanced communities
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure : The matplotlib figure
    """
    # Calculate metrics
    baseline_metrics = calculate_community_metrics(G, baseline_communities)
    enhanced_metrics = calculate_community_metrics(G, enhanced_communities)
    
    baseline_structure = analyze_community_structure(G, baseline_communities)
    enhanced_structure = analyze_community_structure(G, enhanced_communities)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Modularity comparison
    metrics = ['modularity']
    values = [baseline_metrics['modularity'], enhanced_metrics['modularity']]
    axes[0].bar(['Baseline', 'Enhanced'], values, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_title('Modularity Comparison')
    axes[0].set_ylim(max(0, min(values) * 0.9), max(values) * 1.1)
    
    # 2. Community structure
    metrics = ['avg_internal_edge_ratio', 'avg_conductance', 'avg_community_clustering']
    labels = ['Internal Edge Ratio', 'Conductance', 'Clustering']
    
    baseline_values = [baseline_structure['avg_internal_edge_ratio'], 
                      baseline_structure['avg_conductance'],
                      baseline_structure['avg_community_clustering']]
    enhanced_values = [enhanced_structure['avg_internal_edge_ratio'],
                      enhanced_structure['avg_conductance'],
                      enhanced_structure['avg_community_clustering']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, baseline_values, width, label='Baseline', color='#1f77b4')
    axes[1].bar(x + width/2, enhanced_values, width, label='Enhanced', color='#ff7f0e')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_title('Community Structure Metrics')
    axes[1].legend()
    
    # 3. Community size distribution
    baseline_sizes = [len(c) for c in baseline_communities]
    enhanced_sizes = [len(c) for c in enhanced_communities]
    
    axes[2].hist([baseline_sizes, enhanced_sizes], bins=5, 
               label=['Baseline', 'Enhanced'], alpha=0.7,
               color=['#1f77b4', '#ff7f0e'])
    axes[2].set_title('Community Size Distribution')
    axes[2].set_xlabel('Community Size')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    return fig
