"""
Compare our Enhanced Community Detection algorithm with other algorithms.

This script compares multiple community detection algorithms on various networks
and evaluates them based on modularity and other metrics.
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.enhanced_community_detection import EnhancedCommunityDetection
import community as community_louvain  # python-louvain package


def run_algorithm(G, algorithm, **kwargs):
    """
    Run a community detection algorithm and measure performance.
    
    Parameters:
    -----------
    G : nx.Graph
        Input network
    algorithm : str
        Algorithm name
    kwargs : dict
        Additional parameters for the algorithm
        
    Returns:
    --------
    dict : Performance metrics
    """
    start_time = time.time()
    communities = None
    
    if algorithm == 'enhanced':
        # Our enhanced algorithm
        detector = EnhancedCommunityDetection(G)
        detector.detect_baseline_communities()
        communities = detector.enhance_communities(**kwargs)
        modularity = detector.calculate_modularity(communities)
    
    elif algorithm == 'greedy_modularity':
        # Standard greedy modularity optimization
        communities = list(nx.community.greedy_modularity_communities(G))
        modularity = nx.community.modularity(G, communities)
    
    elif algorithm == 'louvain':
        # Louvain algorithm
        partition = community_louvain.best_partition(G)
        # Convert to list of sets format
        comm_dict = {}
        for node, comm_id in partition.items():
            if comm_id not in comm_dict:
                comm_dict[comm_id] = set()
            comm_dict[comm_id].add(node)
        communities = list(comm_dict.values())
        modularity = nx.community.modularity(G, communities)
    
    elif algorithm == 'label_propagation':
        # Label Propagation
        communities = list(nx.algorithms.community.label_propagation_communities(G))
        modularity = nx.community.modularity(G, communities)
    
    elif algorithm == 'fluid':
        # Asynchronous Fluid Communities
        k = kwargs.get('k', min(5, G.number_of_nodes() // 10))
        communities = list(nx.algorithms.community.asyn_fluidc(G, k))
        modularity = nx.community.modularity(G, communities)
    
    execution_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'algorithm': algorithm,
        'modularity': modularity,
        'num_communities': len(communities),
        'execution_time': execution_time,
        'avg_community_size': np.mean([len(c) for c in communities]),
        'min_community_size': min([len(c) for c in communities]),
        'max_community_size': max([len(c) for c in communities])
    }
    
    return metrics, communities


def compare_algorithms(network_name, G, algorithms, output_dir=None):
    """
    Compare multiple algorithms on a single network.
    
    Parameters:
    -----------
    network_name : str
        Name of the network (for reporting)
    G : nx.Graph
        The network to analyze
    algorithms : list of str
        List of algorithm names to compare
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame : Comparison metrics
    dict : Dictionary of communities from each algorithm
    """
    results = []
    all_communities = {}
    
    print(f"Comparing algorithms on {network_name} network")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    for algorithm in algorithms:
        print(f"Running {algorithm}...")
        
        if algorithm == 'enhanced':
            metrics, communities = run_algorithm(G, algorithm, 
                                                clustering_threshold=0.2,
                                                internal_connectivity_threshold=0.3)
        elif algorithm == 'fluid':
            # For fluid, we need to specify k
            k = min(5, G.number_of_nodes() // 10)
            metrics, communities = run_algorithm(G, algorithm, k=k)
        else:
            metrics, communities = run_algorithm(G, algorithm)
            
        results.append(metrics)
        all_communities[algorithm] = communities
        
        print(f"  - Found {len(communities)} communities")
        print(f"  - Modularity: {metrics['modularity']:.4f}")
        print(f"  - Execution time: {metrics['execution_time']:.2f} seconds")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, f"{network_name}_comparison.csv"), index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Plot modularity
        plt.subplot(1, 2, 1)
        plt.bar(results_df['algorithm'], results_df['modularity'])
        plt.title("Modularity Comparison")
        plt.ylabel("Modularity")
        plt.xticks(rotation=45)
        
        # Plot execution time
        plt.subplot(1, 2, 2)
        plt.bar(results_df['algorithm'], results_df['execution_time'])
        plt.title("Execution Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{network_name}_comparison.png"), dpi=300)
    
    return results_df, all_communities


def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'algorithm_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # List of algorithms to compare
    algorithms = ['enhanced', 'greedy_modularity', 'louvain', 'label_propagation', 'fluid']
    
    # Compare on Zachary's Karate Club network
    karate_club = nx.karate_club_graph()
    karate_results, _ = compare_algorithms("karate_club", karate_club, algorithms, output_dir)
    
    # Compare on Les Misérables network
    lesmis = nx.les_miserables_graph()
    lesmis_results, _ = compare_algorithms("les_miserables", lesmis, algorithms, output_dir)
    
    # Generate and compare on a larger synthetic network
    print("\nGenerating synthetic network...")
    synthetic = nx.stochastic_block_model(
        sizes=[20, 20, 20, 20, 20],
        p=[[0.3, 0.02, 0.02, 0.01, 0.01],
           [0.02, 0.3, 0.01, 0.02, 0.01],
           [0.02, 0.01, 0.3, 0.02, 0.01],
           [0.01, 0.02, 0.02, 0.3, 0.02],
           [0.01, 0.01, 0.01, 0.02, 0.3]],
        seed=42
    )
    synthetic_results, _ = compare_algorithms("synthetic", synthetic, algorithms, output_dir)
    
    # Combine all results
    all_results = pd.concat([
        karate_results.assign(network="Karate Club"),
        lesmis_results.assign(network="Les Misérables"),
        synthetic_results.assign(network="Synthetic")
    ])
    
    # Save combined results
    all_results.to_csv(os.path.join(output_dir, "combined_comparison.csv"), index=False)
    
    # Create summary plot
    plt.figure(figsize=(15, 10))
    
    networks = ["Karate Club", "Les Misérables", "Synthetic"]
    
    # Plot modularity by network and algorithm
    plt.subplot(2, 1, 1)
    for i, alg in enumerate(algorithms):
        modularity_values = [
            karate_results[karate_results['algorithm'] == alg]['modularity'].values[0],
            lesmis_results[lesmis_results['algorithm'] == alg]['modularity'].values[0],
            synthetic_results[synthetic_results['algorithm'] == alg]['modularity'].values[0]
        ]
        plt.bar([x + i*0.15 for x in range(len(networks))], 
                modularity_values, width=0.15, label=alg)
    
    plt.title("Modularity by Network and Algorithm")
    plt.xticks([x + 0.3 for x in range(len(networks))], networks)
    plt.ylabel("Modularity")
    plt.legend()
    
    # Plot execution time by network and algorithm
    plt.subplot(2, 1, 2)
    for i, alg in enumerate(algorithms):
        time_values = [
            karate_results[karate_results['algorithm'] == alg]['execution_time'].values[0],
            lesmis_results[lesmis_results['algorithm'] == alg]['execution_time'].values[0],
            synthetic_results[synthetic_results['algorithm'] == alg]['execution_time'].values[0]
        ]
        plt.bar([x + i*0.15 for x in range(len(networks))], 
                time_values, width=0.15, label=alg)
    
    plt.title("Execution Time by Network and Algorithm")
    plt.xticks([x + 0.3 for x in range(len(networks))], networks)
    plt.ylabel("Time (seconds)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison_summary.png"), dpi=300)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
