"""
Generate a political blogs network for demonstration purposes.
"""

import networkx as nx
import numpy as np
import random
import community
import os

def generate_political_blogs_network(n_conservative=50, n_liberal=50, 
                                    p_within=0.15, p_between=0.01, seed=42):
    """Generate a synthetic political blogs network with two communities."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Create empty graph
    G = nx.DiGraph()
    
    # Add conservative blogs (nodes 0 to n_conservative-1)
    for i in range(n_conservative):
        G.add_node(i, value='conservative', label=f"conservative_blog_{i}")
    
    # Add liberal blogs (nodes n_conservative to n_conservative+n_liberal-1)
    for i in range(n_conservative, n_conservative + n_liberal):
        G.add_node(i, value='liberal', label=f"liberal_blog_{i}")
    
    # Add edges within conservative community
    for i in range(n_conservative):
        for j in range(i+1, n_conservative):
            if random.random() < p_within:
                if random.random() < 0.5:  # make some edges bidirectional
                    G.add_edge(i, j)
                    G.add_edge(j, i)
                else:
                    G.add_edge(i, j)
    
    # Add edges within liberal community
    for i in range(n_conservative, n_conservative + n_liberal):
        for j in range(i+1, n_conservative + n_liberal):
            if random.random() < p_within:
                if random.random() < 0.5:  # make some edges bidirectional
                    G.add_edge(i, j)
                    G.add_edge(j, i)
                else:
                    G.add_edge(i, j)
    
    # Add some edges between communities (much fewer)
    for i in range(n_conservative):
        for j in range(n_conservative, n_conservative + n_liberal):
            if random.random() < p_between:
                if random.random() < 0.3:  # make few edges bidirectional
                    G.add_edge(i, j)
                    G.add_edge(j, i)
                else:
                    if random.random() < 0.5:
                        G.add_edge(i, j)
                    else:
                        G.add_edge(j, i)
    
    return G

def save_as_gml(G, filename):
    """Save the network in GML format."""
    # Convert node IDs to strings for GML compatibility
    G_gml = nx.DiGraph()
    
    for node in G.nodes():
        # Add nodes with attributes
        G_gml.add_node(str(node), 
                      label=G.nodes[node].get('label', f"blog_{node}"),
                      value=G.nodes[node].get('value', 'unknown'))
    
    for u, v in G.edges():
        G_gml.add_edge(str(u), str(v))
    
    nx.write_gml(G_gml, filename)
    print(f"Network saved to {filename}")

def main():
    # Generate the network
    print("Generating political blogs network...")
    G = generate_political_blogs_network(n_conservative=60, n_liberal=50, 
                                       p_within=0.1, p_between=0.01)
    
    print(f"Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath("data/external/polblogs_large.gml")), exist_ok=True)
    
    # Save as GML
    save_as_gml(G, "data/external/polblogs_large.gml")
    
    # Create a NetworkX version
    nx.write_gpickle(G, "data/external/polblogs_large.gpickle")

if __name__ == "__main__":
    main()
