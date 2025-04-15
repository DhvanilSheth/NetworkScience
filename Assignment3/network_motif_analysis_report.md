
    # Network Motif Analysis Summary Report
    
    ## Network Statistics
    - Network type: Directed graph
    - Number of nodes: 348
    - Number of edges: 1010
    - Network density: 0.008364
    
    ## Subgraph Analysis
    - Number of unique 3-node connected subgraph patterns: 5
    - Total subgraphs observed in the network: 8851
    - Number of statistically significant motifs (Z > 2.0): 3
    - Number of statistically significant anti-motifs (Z < -2.0): 1
    
    ## Top Motifs (Over-represented Subgraphs)
    
    1. Pattern ID 3:
       - Z-score: 13.77
       - Observed count: 353 (vs. expected 90.90)
       - Ratio: 3.88x more frequent than expected by chance
            
    2. Pattern ID 4:
       - Z-score: 10.12
       - Observed count: 64 (vs. expected 12.20)
       - Ratio: 5.25x more frequent than expected by chance
            
    3. Pattern ID 1:
       - Z-score: 4.77
       - Observed count: 5533 (vs. expected 5090.00)
       - Ratio: 1.09x more frequent than expected by chance
            
    ## Top Anti-Motifs (Under-represented Subgraphs)
    
    1. Pattern ID 0:
       - Z-score: -3.88
       - Observed count: 2897 (vs. expected 3295.30)
       - Ratio: 0.88x less frequent than expected by chance
            
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
    