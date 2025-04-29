"""
Generate synthetic networks with known community structure.

This script generates synthetic networks with known community structures
using stochastic block models.
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

def generate_sbm_network(n=500, k=5, p_in=0.3, p_out=0.01, seed=42):
    """
    Generate a network using the Stochastic Block Model.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    k : int
        Number of communities
    p_in : float
        Probability of edge within communities
    p_out : float
        Probability of edge between communities
    seed : int
        Random seed
        
    Returns:
    --------
    G : nx.Graph
        Generated network
    communities : dict
        Dictionary mapping community IDs to sets of nodes
    """
    np.random.seed(seed)
    
    # Calculate community sizes
    sizes = [n // k] * k
    # Add remaining nodes to last community
    if sum(sizes) < n:
        sizes[-1] += n - sum(sizes)
    
    # Create probability matrix
    p_matrix = np.ones((k, k)) * p_out
    for i in range(k):
        p_matrix[i, i] = p_in
        
    # Generate the network
    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    
    # Create community mapping
    communities = {}
    node_id = 0
    for comm_id, size in enumerate(sizes):
        community = set()
        for _ in range(size):
            G.nodes[node_id]['community'] = comm_id
            community.add(node_id)
            node_id += 1
        communities[comm_id] = community
        
    return G, communities

def save_network_with_communities(G, communities, graph_file, communities_file):
    """
    Save the network and its community structure to files.
    
    Parameters:
    -----------
    G : nx.Graph
        The network
    communities : dict
        Dictionary mapping community labels to sets of nodes
    graph_file : str
        Path where to save the network
    communities_file : str
        Path where to save community memberships
    """
    # Save the network as edgelist
    nx.write_edgelist(G, graph_file, data=False)
    
    # Save community memberships
    community_data = []
    for comm_id, nodes in communities.items():
        for node in nodes:
            community_data.append({
                'Node': node,
                'Community': comm_id
            })
    
    pd.DataFrame(community_data).to_csv(communities_file, index=False)
    
    print(f"Network saved to {graph_file}")
    print(f"Community structure saved to {communities_file}")

if __name__ == "__main__":
    # Get the project root directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_dir, "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating synthetic networks in {output_dir}")
    
    # Generate networks with different mixing levels
    
    # Well-defined communities (high internal density, low external)
    G1, comm1 = generate_sbm_network(n=100, k=5, p_in=0.3, p_out=0.01, seed=42)
    save_network_with_communities(
        G1, comm1,
        os.path.join(output_dir, "sbm_clear_communities.edgelist"),
        os.path.join(output_dir, "sbm_clear_communities.csv")
    )
    
    # Medium-mixed communities
    G2, comm2 = generate_sbm_network(n=100, k=5, p_in=0.2, p_out=0.05, seed=43)
    save_network_with_communities(
        G2, comm2,
        os.path.join(output_dir, "sbm_medium_communities.edgelist"),
        os.path.join(output_dir, "sbm_medium_communities.csv")
    )
    
    # Ambiguous communities (lower internal density, higher external)
    G3, comm3 = generate_sbm_network(n=100, k=5, p_in=0.15, p_out=0.08, seed=44)
    save_network_with_communities(
        G3, comm3,
        os.path.join(output_dir, "sbm_ambiguous_communities.edgelist"),
        os.path.join(output_dir, "sbm_ambiguous_communities.csv")
    )
