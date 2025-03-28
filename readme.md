# Readme

This code runs simulations of a problem called Cost Function Firefighter, a variant of The Firefighter Problem (see
survey by Finbow and MacGillivray [1]). We implement four cost functions (some with variable amount of stochasticity)
and three main heuristics, each of which can be combined with another heuristic to break ties.

Main simulation code can be found in `simulation.py` - the code in the main method can be used to reproduce the results
used in the associated extended manuscript (under review). This relies on soct function and heuristic definitions in `costs_and_heuristics.py`.
The file `prelim_code.py` contains code used to run
preliminary simulations presented in the original conference paper (in press). The file `graph_analysis.py` contains
code used to analyze contact graphs and `plotting.py` contains mostly helper methods to produce the plots used in the
extended publication.

In the output directory, we provide results of simulations used to produce the plots in the extended publication (and
those plots).

## Heuristic strategies

Heuristics create strategies by selecting vertices per turn greedily with respect to:
1. **Highest degree**
2. **Lowest cost**
3. **Highest threat** (proximity to fire)

We also implement a strategy that defends randomly for comparison.

## Cost functions

We implemented the following cost functions:
1. **Uniform** - defaults to 1 for all vertices
2. **Uniformly random** - costs chosen uniformly at random from the specified or default range [1,5]
3. **Binary hesitation** - costs are 2 with probability given probability (defaults to 29.72%, to model average rate of 
   vaccine hesitancy reported by Pourrazavi _et al._ [2]), 1 otherwise
4. **Threat with stochasticity** - costs are proportional to the threat of the vertex (proximity to fire), with random 
   noise added to each cost by adding an integer from the specified range

## About the Network Data Files

Information about the network data files and how they are currently formatted.

### Formatting

Files are saved as CSV files. They can have headings, although the actual names are ignored.

This repo contains two files used in the publication

### Raccoons

- Filename: `mammalia-raccoon-proximity.csv`
- From: [Network Data Repository](https://networkrepository.com/mammalia-raccoon-proximity.php), [original source](https://bansallab.github.io/asnr/data.html)
- Used in citation: Reynolds, Jennifer JH, et al. "Raccoon contact networks predict seasonal susceptibility to rabies
  outbreaks and limitations of vaccination." Journal of Animal Ecology 84.6 (2015): 1720-1731.
- Extra data:
    - Third column: weights for edges
    - Fourth column edge timestamps.
- Nodes: 24
- Edges: 
- Density: 
- Max degree: 
- Min degree: 
- Avg degree: 
- Assortativity:
- 

### Lizard Social Network

Sleepy Lizard population in Australia. Two lizards were assumed to had made a social contact if they were within 2 m of
each other at any of the synchronized
10 min GPS locations.

- Filename: `reptilia-lizard-network-social.csv`
- From: [Network Data Repository](https://networkrepository.com/reptilia-lizard-network-social.php), [original source](https://bansallab.github.io/asnr/data.html)
- Used in citation: Bull, C. M., S. S. Godfrey, and D. M. Gordon. "Social networks and the spread of Salmonella in a
  sleepy lizard population." Molecular Ecology 21.17 (2012): 4386-4392.
    - Abstract: "Although theoretical models consider social networks as pathways for disease transmission, strong
      empirical support, particularly for indirectly transmitted parasites, is lacking for many wildlife populations. We
      found multiple genetic strains of the enteric bacterium _Salmonella enterica_ within a population of Australian
      sleepy lizards (_Tiliqua rugosa_), and we found that pairs of lizards that shared bacterial genotypes were more
      strongly connected in the social network than were pairs of lizards that did not. In contrast, there was no
      significant association between spatial proximity of lizard pairs and shared bacterial genotypes. These results
      provide strong correlative evidence that these bacteria are transmitted from host to host around the social
      network, rather than that adjacent lizards are picking up the same bacterial genotype from some common source."
- Nodes: 60
- Edges: 318
- Density: 0.179661
- Maximum degree: 23
- Minimum degree: 2
- Average degree: 10
- Assortativity: 0.350035


References
----------
[1]: Finbow, S. and MacGillivray, G. _The Firefighter Problem: a survey of results, directions and questions._ 
      Australasian Journal of Combinatorics. 43 (2009): 57-78.

[2]: Pourrazavi S., Fathifar Z., Sharma M., Allahverdipour H. _COVID-19 vaccine hesitancy: A Systematic review of 
      cognitive determinants._ Health Promotion Perspect. 2023 Apr 30;13(1):21-35. doi: 10.34172/hpp.2023.03. 
      PMID: 37309435; PMCID: PMC10257562.