# Causal gene-based ranking

This branch is intended for the code where we rank known causal genes by how likely these were causal for a specific set of SVs and/or SNVs from cancer patients. An idea of the causal SVs and SNVs can then be obtained from the highest ranked genes. 

The currently included datasets are TADs and eQTLs. Hi-C interactions were tested but do not work at the moment. 
 

In the settings file the required file paths can be specified. 

# How to use

All code specific for the causal gene based ranking is in the folder `CausalGeneRanking`. 

The starting script is `runRankingWithPermutations.sh`. This script does not require any parameters. If run on the HPC, it will first score all causal genes for causality and then repeat the process 1000 times with SVs and/or SNVs permuted across the genome. 

If these scripts are done, the `computePValuesPerGene.py` script can be run to do the actual ranking of the genes. As parameters the script requires the output folder containing the scores for the normal run and
1000 permutations (folder name chosen at random for the time being, within the `RankedGenes` folder), and the number of permutations that were run (+1 because there is currently still a bug :)). So:
```
computePValuesPerGene.py RankedGenes/"outputFolderNameChosenAtRandom" 1000 
```
The output is a list of all causal genes that have significant p-values in as much layers as possible.

To test without all permutations and just doing the initial scoring of genes, run:
```
main.py "runName" N
```
For example,

```
main.py ABC N
```
will run the code without permutations (N for no) and will write the output to folder `ABC` within the `RankedGenes` folder. 

