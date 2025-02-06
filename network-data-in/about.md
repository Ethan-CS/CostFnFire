# About the Network Data Files

Extra information about the network data files and how they are currently formatted.

## Formatting

Files are saved as CSV files. They can have headings, although the actual names are ignored.

## Raccoons

- Filename: `mammalia-raccoon-proximity.csv`
-
From: [Network Data Repository](https://networkrepository.com/mammalia-raccoon-proximity.php), [original source](https://bansallab.github.io/asnr/data.html)
- Used in citation: Reynolds, Jennifer JH, et al. "Raccoon contact networks predict seasonal susceptibility to rabies
  outbreaks and limitations of vaccination." Journal of Animal Ecology 84.6 (2015): 1720-1731.
- Extra data:
    - Third column: weights for edges
    - Fourth column edge timestamps.
- Nodes: 24
- Edges: 2K
- Density: 7.23551
- Max degree: 280
- Min degree: 14
- Avg degree: 166
- Assortativity: 0.0761359

## Tortoises

- Filename: `reptilia-tortoise-network-fi.csv`
-
From: [Network Data Repository](https://networkrepository.com/reptilia-tortoise-network-fi.php), [original source](https://bansallab.github.io/asnr/data.html)
- Used in citation: Sah, Pratha et al. "Inferring social structure and its drivers from refuge use in the desert
  tortoise, a relatively solitary species." Behavioral Ecology and Sociobiology 70.8 (2016): 1277-1289.
- Extra data:
    - Third column: edge timestamps
- Nodes: 787
- Edges: 1.7K
- Density: 0.00553847
- Max degree: 40
- Min degree: 1
- Average degree: 4
- Assortativity: 0.53928

## Malawi village

- Filename: `social-malawi-village.csv`
- From: [SocioPatterns](http://www.sociopatterns.org/datasets/contact-patterns-in-a-village-in-rural-malawi/)
- Used in citation: L. Ozella et al., “Using wearable proximity sensors to characterize social contact patterns in a
  village of rural Malawi”, EPJ Data Science 10, 46 (2021).
- Nodes: 86
- Edges: 347
- Density: 0.09493844049247606
- Max degree: 31
- Min degree: 1
- Average degree: 8.069767441860465
- Assortativity: 0.03625990752158057

## Lizard Social Network

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
