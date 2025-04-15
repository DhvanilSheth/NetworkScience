
    # Network Motif Analysis Summary Report
    
    ## Network Statistics
    - Network type: Directed graph
    - Number of nodes: 312
    - Number of edges: 817
    - Network density: 0.008420
    
    ## Subgraph Analysis
    - Number of unique 3-node connected subgraph patterns: 5
    - Total subgraphs observed in the network: 4783
    - Number of statistically significant motifs (Z > 2.0): 2
    - Number of statistically significant anti-motifs (Z < -2.0): 1
    
    ## Top Motifs (Over-represented Subgraphs)
    
    1. Pattern ID 4:
       - Z-score: 10.61
       - Observed count: 43 (vs. expected 4.10)
       - Ratio: 10.49x more frequent than expected by chance
            
    2. Pattern ID 3:
       - Z-score: 4.57
       - Observed count: 169 (vs. expected 59.60)
       - Ratio: 2.84x more frequent than expected by chance
            
    ## Top Anti-Motifs (Under-represented Subgraphs)
    
    1. Pattern ID 0:
       - Z-score: -4.83
       - Observed count: 1582 (vs. expected 2082.90)
       - Ratio: 0.76x less frequent than expected by chance
            
    ## Conclusions
    
    1. **Structural Organization**: The network shows clear evidence of non-random organization at the local level. 
       Specific 3-node patterns appear more or less frequently than chance expectations.
    
    2. **Citation Patterns (if this is a citation network)**: Over-represented motifs can indicate 
       characteristic citation behaviors (hierarchical, reciprocal, or cyclical). 
       Under-represented subgraphs may reflect disfavored structures.
    
    3. **Random Models**: Results depend on the random model used (configuration vs. edge-swap). 
       The edge-swap model preserves in/out degree but can break certain structural correlations.
    
    4. **Functional Significance**: Over-represented motifs (motifs) often provide functional advantages (robustness, 
       efficient information flow), while anti-motifs might be inefficient or destabilizing patterns.
    