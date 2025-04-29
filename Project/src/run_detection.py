"""
Command-line interface for Enhanced Community Detection algorithm.
"""

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_community_detection import EnhancedCommunityDetection
from src.data_utils import load_network

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Community Detection")
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input network file')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    parser.add_argument('--directed', action='store_true',
                        help='Whether the network is directed')
    
    parser.add_argument('--weighted', action='store_true',
                        help='Whether the network has edge weights')
    
    parser.add_argument('--clustering-threshold', type=float, default=0.2,
                        help='Threshold for clustering coefficient')
    
    parser.add_argument('--connectivity-threshold', type=float, default=0.3,
                        help='Threshold for internal connectivity')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading network from {args.input}...")
    G = load_network(args.input, directed=args.directed, weighted=args.weighted)
    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the detector
    print("Initializing community detector...")
    detector = EnhancedCommunityDetection(G)
    
    # Detect baseline communities
    print("Running baseline community detection...")
    start_time = time.time()
    baseline_communities = detector.detect_baseline_communities()
    baseline_time = time.time() - start_time
    print(f"Detected {len(baseline_communities)} communities in {baseline_time:.2f} seconds.")
    
    # Run enhanced detection
    print("Running enhanced community detection...")
    start_time = time.time()
    enhanced_communities = detector.enhance_communities(
        clustering_threshold=args.clustering_threshold,
        internal_connectivity_threshold=args.connectivity_threshold
    )
    enhanced_time = time.time() - start_time
    print(f"Detected {len(enhanced_communities)} communities in {enhanced_time:.2f} seconds.")
    
    # Calculate modularity
    baseline_modularity = detector.calculate_modularity(baseline_communities)
    enhanced_modularity = detector.calculate_modularity(enhanced_communities)
    print(f"Baseline Modularity: {baseline_modularity:.4f}")
    print(f"Enhanced Modularity: {enhanced_modularity:.4f}")
    print(f"Improvement: {enhanced_modularity - baseline_modularity:.4f}")
    
    # Calculate metrics
    metrics = detector.calculate_community_metrics()
    metrics_file = os.path.join(args.output, 'community_metrics.csv')
    metrics.to_csv(metrics_file, index=False)
    print(f"Saved community metrics to {metrics_file}")
    
    # Identify reassigned nodes
    reassigned_nodes = []
    for node in G.nodes():
        if (node in detector.node_to_community_map_baseline and 
            node in detector.node_to_community_map_enhanced and
            detector.node_to_community_map_baseline[node] != detector.node_to_community_map_enhanced[node]):
            
            reassigned_nodes.append({
                'Node': node,
                'Original_Community': detector.node_to_community_map_baseline[node],
                'New_Community': detector.node_to_community_map_enhanced[node],
                'Degree': G.degree(node),
                'Clustering': detector.clustering_coefficients[node]
            })
    
    if reassigned_nodes:
        reassigned_df = pd.DataFrame(reassigned_nodes)
        reassigned_file = os.path.join(args.output, 'reassigned_nodes.csv')
        reassigned_df.to_csv(reassigned_file, index=False)
        print(f"Saved {len(reassigned_nodes)} reassigned nodes to {reassigned_file}")
    else:
        print("No nodes were reassigned between communities.")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        
        # Community visualization
        fig, axes = detector.visualize_communities(method='both', figsize=(16, 8))
        vis_file = os.path.join(args.output, 'community_visualization.png')
        fig.savefig(vis_file, dpi=300, bbox_inches='tight')
        print(f"Saved community visualization to {vis_file}")
        
        # Save communities to files
        baseline_comm_file = os.path.join(args.output, 'baseline_communities.csv')
        enhanced_comm_file = os.path.join(args.output, 'enhanced_communities.csv')
        
        # Convert communities to DataFrames
        baseline_comm_data = []
        for i, comm in enumerate(baseline_communities):
            for node in comm:
                baseline_comm_data.append({'Node': node, 'Community': i})
        
        enhanced_comm_data = []
        for i, comm in enumerate(enhanced_communities):
            for node in comm:
                enhanced_comm_data.append({'Node': node, 'Community': i})
        
        pd.DataFrame(baseline_comm_data).to_csv(baseline_comm_file, index=False)
        pd.DataFrame(enhanced_comm_data).to_csv(enhanced_comm_file, index=False)
        
        print(f"Saved community assignments to {baseline_comm_file} and {enhanced_comm_file}")
    
    print("Done!")

if __name__ == '__main__':
    main()
