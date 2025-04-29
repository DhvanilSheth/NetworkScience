"""
Enhanced Modularity-Based Community Detection

This module implements the enhanced community detection algorithm that 
combines Greedy Modularity Optimization with local structure analysis.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import community as community_louvain
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional


class EnhancedCommunityDetection:
    """
    Enhanced Community Detection algorithm that improves upon Greedy Modularity Optimization
    by incorporating local clustering coefficients and network structural metrics.
    """
    
    def __init__(self, G: nx.Graph):
        """
        Initialize with a networkx graph.
        
        Parameters:
        -----------
        G : nx.Graph
            The input network for community detection
        """
        self.G = G.copy()
        self.original_G = G.copy()
        self.baseline_communities = None
        self.enhanced_communities = None
        self.node_to_community_map_baseline = {}
        self.node_to_community_map_enhanced = {}
        
        # Calculate global network metrics
        self.clustering_coefficients = nx.clustering(self.G)
        self.avg_clustering = nx.average_clustering(self.G)
        self.assortativity = nx.degree_assortativity_coefficient(self.G)
        
    def analyze_network_structure(self) -> Dict[str, Any]:
        """
        Analyze the network structure and return key metrics.
        
        Returns:
        --------
        dict : Dictionary of network metrics
        """
        metrics = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'avg_clustering': self.avg_clustering,
            'assortativity': self.assortativity,
            'avg_path_length': nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None,
            'diameter': nx.diameter(self.G) if nx.is_connected(self.G) else None,
        }
        
        # Calculate degree statistics
        degrees = [d for _, d in self.G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = max(degrees)
        metrics['min_degree'] = min(degrees)
        metrics['degree_std'] = np.std(degrees)
        
        return metrics
    
    def detect_baseline_communities(self) -> List[Set[int]]:
        """
        Detect communities using the baseline Greedy Modularity Optimization method.
        
        Returns:
        --------
        list : List of sets, where each set contains the nodes in a community
        """
        # Use networkx's greedy modularity communities
        self.baseline_communities = list(nx.community.greedy_modularity_communities(self.G))
        
        # Create node-to-community mapping
        for i, community in enumerate(self.baseline_communities):
            for node in community:
                self.node_to_community_map_baseline[node] = i
        
        return self.baseline_communities
    
    def identify_misfit_nodes(self, clustering_threshold: float = 0.2, 
                              internal_connectivity_threshold: float = 0.3) -> List[int]:
        """
        Identify nodes that might be misplaced in their communities.
        
        Parameters:
        -----------
        clustering_threshold : float
            Threshold for considering a node to have low clustering
        internal_connectivity_threshold : float
            Threshold for considering a node to have low internal connectivity
            
        Returns:
        --------
        list : List of potentially misplaced nodes
        """
        if not self.baseline_communities:
            self.detect_baseline_communities()
            
        misfit_nodes = []
        
        for node in self.G.nodes():
            # Skip nodes with very low degree
            if self.G.degree(node) < 3:
                continue
                
            # Check if clustering coefficient is low
            if self.clustering_coefficients[node] < clustering_threshold:
                # Check internal vs external connectivity
                community_idx = self.node_to_community_map_baseline[node]
                current_community = self.baseline_communities[community_idx]
                
                internal_edges = sum(1 for neighbor in self.G.neighbors(node) 
                                   if neighbor in current_community)
                total_edges = self.G.degree(node)
                
                internal_ratio = internal_edges / total_edges if total_edges > 0 else 0
                
                if internal_ratio < internal_connectivity_threshold:
                    misfit_nodes.append(node)
        
        return misfit_nodes
    
    def find_best_community(self, node: int) -> int:
        """
        Find the most suitable community for a given node.
        
        Parameters:
        -----------
        node : int
            The node to reassign
            
        Returns:
        --------
        int : The index of the best community for the node
        """
        best_community_idx = self.node_to_community_map_baseline[node]
        best_score = -1
        
        # Get neighbors of the node
        neighbors = list(self.G.neighbors(node))
        
        # Create a counter for neighbor communities
        neighbor_communities = Counter([self.node_to_community_map_baseline[neigh] 
                                      for neigh in neighbors])
        
        # Calculate a connectivity score for each community
        for comm_idx, count in neighbor_communities.items():
            community = self.baseline_communities[comm_idx]
            
            # Calculate weighted score based on:
            # 1. Number of neighbors in this community
            # 2. Weighted by edge weights if available
            edge_weight_sum = 0
            for neigh in neighbors:
                if self.node_to_community_map_baseline[neigh] == comm_idx:
                    if 'weight' in self.G[node][neigh]:
                        edge_weight_sum += self.G[node][neigh]['weight']
                    else:
                        edge_weight_sum += 1
            
            # Calculate combined score
            connectivity_score = count / len(neighbors) if neighbors else 0
            score = 0.7 * connectivity_score + 0.3 * (edge_weight_sum / self.G.degree(node, weight='weight'))
            
            if score > best_score:
                best_score = score
                best_community_idx = comm_idx
                
        return best_community_idx
    
    def enhance_communities(self, clustering_threshold: float = 0.2, 
                           internal_connectivity_threshold: float = 0.3) -> List[Set[int]]:
        """
        Enhance the community detection by reassigning misfit nodes.
        
        Parameters:
        -----------
        clustering_threshold : float
            Threshold for considering a node to have low clustering
        internal_connectivity_threshold : float
            Threshold for considering a node to have low internal connectivity
            
        Returns:
        --------
        list : List of enhanced communities
        """
        if not self.baseline_communities:
            self.detect_baseline_communities()
            
        # Start with baseline communities
        self.enhanced_communities = [set(comm) for comm in self.baseline_communities]
        self.node_to_community_map_enhanced = self.node_to_community_map_baseline.copy()
        
        # Identify misfit nodes
        misfit_nodes = self.identify_misfit_nodes(
            clustering_threshold, internal_connectivity_threshold
        )
        
        # Reassign misfit nodes
        for node in misfit_nodes:
            current_comm_idx = self.node_to_community_map_enhanced[node]
            best_comm_idx = self.find_best_community(node)
            
            # If best community is different, reassign
            if best_comm_idx != current_comm_idx:
                self.enhanced_communities[current_comm_idx].remove(node)
                self.enhanced_communities[best_comm_idx].add(node)
                self.node_to_community_map_enhanced[node] = best_comm_idx
        
        # Remove any empty communities
        self.enhanced_communities = [comm for comm in self.enhanced_communities if len(comm) > 0]
        
        # Rebuild the node-to-community mapping
        self.node_to_community_map_enhanced = {}
        for i, community in enumerate(self.enhanced_communities):
            for node in community:
                self.node_to_community_map_enhanced[node] = i
        
        return self.enhanced_communities
    
    def calculate_modularity(self, communities: List[Set[int]] = None) -> float:
        """
        Calculate the modularity of a given community structure.
        
        Parameters:
        -----------
        communities : list of sets, optional
            The community structure to evaluate. If None, uses enhanced_communities.
            
        Returns:
        --------
        float : The modularity score
        """
        if communities is None:
            if self.enhanced_communities:
                communities = self.enhanced_communities
            elif self.baseline_communities:
                communities = self.baseline_communities
            else:
                raise ValueError("No communities available")
                
        # Convert communities to format expected by modularity function
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
                
        return nx.community.modularity(self.G, communities)
    
    def calculate_community_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for comparing baseline and enhanced communities.
        
        Returns:
        --------
        pd.DataFrame : DataFrame comparing metrics for both methods
        """
        if not self.baseline_communities or not self.enhanced_communities:
            raise ValueError("Both baseline and enhanced communities must be detected first")
            
        metrics = {
            'Method': ['Baseline', 'Enhanced'],
            'Modularity': [
                self.calculate_modularity(self.baseline_communities),
                self.calculate_modularity(self.enhanced_communities)
            ],
            'Number of Communities': [
                len(self.baseline_communities),
                len(self.enhanced_communities)
            ]
        }
        
        # Calculate average community size
        baseline_sizes = [len(comm) for comm in self.baseline_communities]
        enhanced_sizes = [len(comm) for comm in self.enhanced_communities]
        
        metrics['Avg Community Size'] = [
            np.mean(baseline_sizes),
            np.mean(enhanced_sizes)
        ]
        
        metrics['Community Size Std'] = [
            np.std(baseline_sizes),
            np.std(enhanced_sizes)
        ]
        
        # Calculate average internal density
        baseline_density = self._calculate_avg_internal_density(self.baseline_communities)
        enhanced_density = self._calculate_avg_internal_density(self.enhanced_communities)
        
        metrics['Avg Internal Density'] = [baseline_density, enhanced_density]
        
        # Calculate number of nodes reassigned
        if hasattr(self, 'node_to_community_map_baseline') and hasattr(self, 'node_to_community_map_enhanced'):
            reassigned_count = sum(1 for node in self.G.nodes() 
                                 if (node in self.node_to_community_map_baseline and 
                                     node in self.node_to_community_map_enhanced and
                                     self.node_to_community_map_baseline[node] != self.node_to_community_map_enhanced[node]))
            metrics['Nodes Reassigned'] = [0, reassigned_count]
        
        return pd.DataFrame(metrics)
    
    def _calculate_avg_internal_density(self, communities: List[Set[int]]) -> float:
        """
        Calculate the average internal density of communities.
        
        Parameters:
        -----------
        communities : list of sets
            List of communities (sets of nodes)
            
        Returns:
        --------
        float : Average internal density
        """
        internal_densities = []
        
        for community in communities:
            if len(community) < 2:
                continue
                
            subgraph = self.G.subgraph(community)
            max_edges = len(community) * (len(community) - 1) / 2
            actual_edges = subgraph.number_of_edges()
            internal_densities.append(actual_edges / max_edges if max_edges > 0 else 0)
            
        return np.mean(internal_densities) if internal_densities else 0
    
    def visualize_communities(self, method: str = 'both', 
                            layout: Optional[Dict] = None,
                            figsize: Tuple[int, int] = (16, 8)) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Visualize communities.
        
        Parameters:
        -----------
        method : str
            'baseline', 'enhanced', or 'both' to determine which communities to plot
        layout : dict, optional
            Node positions for the visualization. If None, uses spring layout.
        figsize : tuple
            Figure size for the plot
            
        Returns:
        --------
        Tuple : (fig, axes) containing the matplotlib figure and axes objects
        """
        if method not in ['baseline', 'enhanced', 'both']:
            raise ValueError("Method must be 'baseline', 'enhanced', or 'both'")
            
        if method == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = axes
        else:
            fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
            if method == 'baseline':
                ax1 = ax
            else:
                ax2 = ax
                
        # Generate a layout if not provided
        if layout is None:
            layout = nx.spring_layout(self.G)
            
        # Define a colormap for the communities
        cmap = plt.cm.get_cmap('tab20', max(
            len(self.baseline_communities) if self.baseline_communities else 0,
            len(self.enhanced_communities) if self.enhanced_communities else 0
        ))
            
        # Plot baseline communities if requested
        if method in ['baseline', 'both']:
            if self.baseline_communities:
                self._draw_communities(ax1, self.baseline_communities, 
                                      self.node_to_community_map_baseline, layout, cmap)
                ax1.set_title("Baseline Communities")
            
        # Plot enhanced communities if requested
        if method in ['enhanced', 'both']:
            if self.enhanced_communities:
                self._draw_communities(ax2, self.enhanced_communities, 
                                      self.node_to_community_map_enhanced, layout, cmap)
                ax2.set_title("Enhanced Communities")
                
        plt.tight_layout()
        return fig, [ax1] if method == 'baseline' else ([ax2] if method == 'enhanced' else [ax1, ax2])
    
    def _draw_communities(self, ax, communities, node_to_comm, layout, cmap):
        """Helper method for drawing communities on an axis."""
        # Draw nodes
        for node in self.G.nodes():
            if node in node_to_comm:
                comm_idx = node_to_comm[node]
                ax.scatter(layout[node][0], layout[node][1], 
                           c=[cmap(comm_idx)], s=100, edgecolors='w')
                
        # Draw edges
        for u, v in self.G.edges():
            ax.plot([layout[u][0], layout[v][0]], 
                    [layout[u][1], layout[v][1]], 
                    'k-', alpha=0.1)
            
        ax.axis('off')
    
    def infer_node_domains(self, num_domains: int = 2) -> Dict[int, str]:
        """
        Infer potential domains of nodes based on community structure.
        This is a simplified approach that assumes nodes in the same community
        likely belong to similar domains.
        
        Parameters:
        -----------
        num_domains : int
            Number of domains to infer
            
        Returns:
        --------
        dict : Mapping from node to inferred domain
        """
        if not self.enhanced_communities:
            self.enhance_communities()
            
        # Create a feature vector for each node based on community structure
        features = {}
        
        # For each node, create a feature based on its community and neighborhood
        for node in self.G.nodes():
            if node in self.node_to_community_map_enhanced:
                comm_idx = self.node_to_community_map_enhanced[node]
                neighbors = list(self.G.neighbors(node))
                
                # Count neighbors in each community
                comm_counts = Counter([self.node_to_community_map_enhanced.get(n, -1) 
                                     for n in neighbors])
                
                # Create a simple feature vector: [degree, clustering_coef, comm_idx]
                features[node] = [
                    self.G.degree(node),
                    self.clustering_coefficients[node],
                    comm_idx,
                    len(comm_counts) / len(self.enhanced_communities) if self.enhanced_communities else 0,
                    comm_counts.get(comm_idx, 0) / len(neighbors) if neighbors else 0
                ]
        
        # Convert to numpy array for clustering
        nodes = list(features.keys())
        feature_matrix = np.array([features[n] for n in nodes])
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        feature_matrix = StandardScaler().fit_transform(feature_matrix)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_domains, random_state=42)
        domains = kmeans.fit_predict(feature_matrix)
        
        # Map cluster labels to domain names (placeholder names)
        domain_names = [f"Domain_{i+1}" for i in range(num_domains)]
        node_domains = {nodes[i]: domain_names[domains[i]] for i in range(len(nodes))}
        
        return node_domains
