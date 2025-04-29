"""
Generate synthetic LFR benchmark network with planted communities.

This script generates a synthetic network with known community structure
using the LFR (Lancichinetti-Fortunato-Radicchi) benchmark model.
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import LFR benchmark if available
try:
    from networkx.generators.community import LFR_benchmark_graph
    has_lfr = True
except ImportError:
    has_lfr = False
    print("NetworkX LFR benchmark not available, using alternative approach")

def create_lfr_network(n=500, tau1=3, tau2=1.5, mu=0.1, average_degree=10, 
                      min_community=20, max_community=50, seed=42):
    """
    Generate an LFR benchmark network with known community structure.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    tau1 : float
        Power law exponent for degree distribution
    tau2 : float
        Power law exponent for community size distribution
    mu : float
        Mixing parameter (fraction of neighbors from different communities)
    average_degree : int
        Average degree of nodes
    min_community : int
        Minimum community size
    max_community : int
        Maximum community size
    seed : int
        Random seed
        
    Returns:
    --------
    G : nx.Graph
        The generated network
    communities : dict
        Dictionary mapping community labels to sets of nodes
    """
    if has_lfr:
        # Generate LFR benchmark using NetworkX implementation
        G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=average_degree,
                               min_community=min_community, max_community=max_community,
                               seed=seed)
        
        # Extract ground truth communities
        communities = {}
        for node in G.nodes():
            comm_id = G.nodes[node]['community']
            if isinstance(comm_id, list) or isinstance(comm_id, set):
                # Handle overlapping communities (if any)
                for c in comm_id:
                    if c not in communities:
                        communities[c] = set()
                    communities[c].add(node)
            else:
                # Non-overlapping communities
                if comm_id not in communities:
                    communities[comm_id] = set()
                communities[comm_id].add(node)
                
    else:
        # Alternative: Create a network with planted communities
        n_communities = max(5, min(10, int(n / min_community)))
        max_nodes_per_community = min(max_community, int(n / n_communities))
        min_nodes_per_community = min_community
        
        # Initialize empty graph and communities
        G = nx.Graph()
        communities = {}
        node_id = 0
        
        # Generate community sizes following power law
        community_sizes = []
        remaining_nodes = n
        for i in range(n_communities - 1):
            # Generate community size following truncated power law
            if remaining_nodes <= min_nodes_per_community:
                size = remaining_nodes
            else:
                # Simple approximation of power law distribution
                size = min(
                    max(
                        min_nodes_per_community,
                        int(np.random.pareto(tau2 - 1) * min_nodes_per_community)
                    ),
                    min(max_nodes_per_community, remaining_nodes - min_nodes_per_community)
                )
            community_sizes.append(size)
            remaining_nodes -= size
        
        # Add remaining nodes to last community
        if remaining_nodes > 0:
            community_sizes.append(remaining_nodes)
            
        # Create communities
        for comm_id, size in enumerate(community_sizes):
            community = set()
            for _ in range(size):
                G.add_node(node_id, community=comm_id)
                community.add(node_id)
                node_id += 1
            communities[comm_id] = community
        
        # Add edges based on mixing parameter
        # 1. Connect nodes within communities (1-mu fraction of edges)
        target_edges_within = int((1 - mu) * average_degree * n / 2)
        edges_added_within = 0
        
        for comm_id, nodes in communities.items():
            nodes_list = list(nodes)
            # Preferential attachment within community to create hubs
            degrees = {node: 1 for node in nodes_list}  # Initialize with degree 1
            
            while edges_added_within < target_edges_within:
                # Select source node with probability proportional to degree
                source = np.random.choice(nodes_list, p=[degrees[n]/sum(degrees.values()) for n in nodes_list])
                
                # Select target from same community, excluding self and existing connections
                possible_targets = [n for n in nodes_list if n != source and not G.has_edge(source, n)]
                if not possible_targets:
                    break
                
                target = np.random.choice(possible_targets)
                G.add_edge(source, target)
                
                # Update degrees
                degrees[source] += 1
                degrees[target] += 1
                edges_added_within += 1
                
                if edges_added_within >= target_edges_within:
                    break
        
        # 2. Connect nodes between communities (mu fraction of edges)
        target_edges_between = int(mu * average_degree * n / 2)
        edges_added_between = 0
        
        all_nodes = list(G.nodes())
        while edges_added_between < target_edges_between:
            # Select random source node
            source = np.random.choice(all_nodes)
            source_comm = G.nodes[source]['community']
            
            # Select target from different community
            other_comm_nodes = [n for n in all_nodes 
                              if G.nodes[n]['community'] != source_comm and not G.has_edge(source, n)]
            if not other_comm_nodes:
                continue
                
            target = np.random.choice(other_comm_nodes)
            G.add_edge(source, target)
            edges_added_between += 1
            
            if edges_added_between >= target_edges_between:
                break
    
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

def main():
    # Generate and save a low-mixing network (well-separated communities)
    G_low, comm_low = create_lfr_network(n=500, mu=0.1, seed=42)
    save_network_with_communities(
        G_low, comm_low, 
        'data/synthetic/lfr_low_mixing.edgelist',
        'data/synthetic/lfr_low_mixing_communities.csv'
    )
    
    # Generate and save a medium-mixing network (moderately-separated communities)
    G_med, comm_med = create_lfr_network(n=500, mu=0.3, seed=43)
    save_network_with_communities(
        G_med, comm_med, 
        'data/synthetic/lfr_medium_mixing.edgelist',
        'data/synthetic/lfr_medium_mixing_communities.csv'
    )
    
    # Generate and save a high-mixing network (ambiguous communities)
    G_high, comm_high = create_lfr_network(n=500, mu=0.5, seed=44)
    save_network_with_communities(
        G_high, comm_high, 
        'data/synthetic/lfr_high_mixing.edgelist',
        'data/synthetic/lfr_high_mixing_communities.csv'
    )

if __name__ == '__main__':
    main()
