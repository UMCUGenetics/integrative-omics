#Imports
from __future__ import division
import numpy as np
from giggle import Giggle
import math
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE

#Loading or creating the index
index = Giggle.create("index", "bed_files/*.bed.gz") #or index = Giggle("index")
index.files

#Function to read the true negative and true positives sets (data/TN_127.txt and data/TP_127.txt). Returns them as numpy arrays. 
def read_sv_data(data, pathogenic):
    SVdata = []

    with open(data) as f:
        linecount = 0
        #skipping the header
        for line in f:

            if linecount < 1:                                       
                linecount += 1
                continue

            line = line.strip()
            splitline = line.split("\t")

            if pathogenic == 0:                                    
                #Get the chromsome, start and end positions and sv type
                chr1 = str(splitline[0])                        
                start = int(splitline[1])                                  
                chr2 = str(splitline[2])                        
                end = int(splitline[3])                                                
                sv_type = str(splitline[4])
                
                #Add 0 as identifier for negative SVs
                SVdata.append([chr1, start, chr2, end, sv_type, pathogenic])    
        
            elif pathogenic == 1:
                chr1 = "chr" + str(splitline[0])
                chr2 = "chr" + str(splitline[1])
                start = int(splitline[2])       
                end = int(splitline[3])
                sv_type = str(splitline[4])
                
                #Add 1 as identifier for positive SVs
                SVdata.append([chr1, start, chr2, end, sv_type, pathogenic])
    
    SVdata = np.array(SVdata, dtype = "object")                     
    return SVdata

'''
Function to annotate the SVs using Giggle. Translocations are annotated at the breakpoints only in a small surrounding window while duplications and deletions are 
annotated along their entire interval. The annotations are: length, overlapping tads, overlapping genes, gene names, pli scores, number of pli scores > 0.9, rvis scores,
number of rvis scores < 10, chromatin state at breakpoint 1, chromatin state at breakpoint 2. 
'''
def annotate(data, index):
    start_time = time.time()
    breakpoint_window = 100 #Window for determining the overlapping TADs and genes at breakpoints for translocations only, not deletions and duplications
    annotated_SVs = []
    
    for sv in data:
        #Make new list to append annotations to
        current_sv = [sv[0], sv[1], sv[2], sv[3], sv[4], sv[5]]
        genes = ""
        pli_scores = ""
        pli_count = 0
        rvis_scores = ""
        rvis_count = 0
        total_num_HPO_terms = 0
        
        #DELETIONS AND DUPLICATIONS
        if current_sv[4] == "Duplication" or current_sv[4] == "Deletion" or current_sv[4] == "DUP" or current_sv[4] == "DEL": 
        
            #take into account that it is possible that some start and end position are reversed and Giggle cant search this
            if sv[1] <= sv[3]: 
                sv_length = sv[3] - sv[1]
                #search the enitre index using the current chromosome, start position and end position
                result = index.query(current_sv[0], current_sv[1], current_sv[3])
            
            elif sv[1] > sv[3]:
                sv_length = sv[1] - sv[3]
                #search the enitre index using the current chromosome, start position and end position
                result = index.query(current_sv[0], current_sv[3], current_sv[1])
            
            #Total number of TADs overlapping with the SV interval
            tad_hits = result.n_hits(1) #this returns the number of hits in the second file in the index (TADs)
            
            #Total number of genes overlapping with the SV interval
            gene_hits = result.n_hits(0) #same but for genes
            
            #Extracting the gene names and scores from the hits. Order of genes names corresponds with order of scores
            for hit in result[0]: #Iterating over all gene hits
                gene = hit.split()[3] #Extracting the gene name
                genes += (gene + " ") 

                pli = hit.split()[6] #Extracting the pli score
                pli_scores += (pli + " ")
                #If the pli score is larger than 0.9, count it
                if pli != "NA":
                    if float(pli) >= 0.9:
                        pli_count += 1
                    
                rvis = hit.split()[7]
                rvis_scores += (rvis + " ")
                if rvis != "NA":
                    if float(rvis) < 10:
                        rvis_count += 1
                
                #Number of HPO terms associated with the overlapping gene
                num_HPO_terms = int(hit.split("\t")[10])
                total_num_HPO_terms += num_HPO_terms
            
            #if no genes or scores are found to be overlapping, put none as annotation to prevent empty annotations
            if len(genes) == 0: 
                genes += ""
            if len(pli_scores) == 0:
                pli_scores += ""
            if len(rvis_scores) == 0:
                rvis_scores += ""
            
            #Append the annotations to the list of the current SV
            current_sv.append(sv_length)
            current_sv.append(tad_hits)
            current_sv.append(gene_hits)
            current_sv.append(genes)
            current_sv.append(pli_scores)
            current_sv.append(pli_count)
            current_sv.append(rvis_scores)
            current_sv.append(rvis_count)
            current_sv.append(total_num_HPO_terms)
            
        #TRANSLOCATIONS        
        else:
            #the length of the SVs is put to zero (translocation have no easy to define lentgh)
            sv_length = 0
            
            #Hits are determined for genes and TADs seperately because they use different windows. Also, both breakpoints are done seperately 
            result_bp1 = index.query(current_sv[0], current_sv[1] - breakpoint_window, current_sv[1] + breakpoint_window)
            #Number of genes overlapping with the window around breakpoint 1
            gene_hit_bp1 = result_bp1.n_hits(0)
            #Number of TADs overlapping with the window around breakpoint 1
            tad_hit_bp1 = result_bp1.n_hits(1)
            
            #Extracting the gene names and scores from the hits. Order of genes names corresponds with order of scores
            for hit in result_bp1[0]:
                gene = hit.split()[3]
                genes += (gene + " ")

                pli = hit.split()[6]
                pli_scores += (pli + " ")
                if pli != "NA":
                    if float(pli) >= 0.9:
                        pli_count += 1
                    
                rvis = hit.split()[7]
                rvis_scores += (rvis + " ")
                if rvis != "NA":
                    if float(rvis) < 10:
                        rvis_count += 1
                
                num_HPO_terms = int(hit.split("\t")[10])
                total_num_HPO_terms += num_HPO_terms
            
            #Now the same for breakpoint 2
            result_bp2 = index.query(current_sv[0], current_sv[3] - breakpoint_window, current_sv[3] + breakpoint_window)
            #Number of genes overlapping with the window around breakpoint 2
            gene_hit_bp2 = result_bp2.n_hits(0)
            #Number of TADs overlapping with the window around breakpoint 1
            tad_hit_bp2 = result_bp2.n_hits(1)
            
            #Extracting the gene names and scores from the hits. Order of genes names corresponds with order of scores
            for hit in result_bp2[0]:
                gene = hit.split()[3]
                genes += (gene + " ")

                pli = hit.split()[6]
                pli_scores += (pli + " ")
                if pli != "NA":
                    if float(pli) >= 0.9:
                        pli_count += 1
                
                rvis = hit.split()[7]
                rvis_scores += (rvis + " ")
                if rvis != "NA":
                    if float(rvis) < 10:
                        rvis_count += 1
                
                num_HPO_terms = int(hit.split("\t")[10])
                total_num_HPO_terms += num_HPO_terms
            
            #Append all the annotations to the translocations in the same order as the deletions and duplications
            current_sv.append(sv_length)
            current_sv.append(tad_hit_bp1 + tad_hit_bp2)
            current_sv.append(gene_hit_bp1 + gene_hit_bp2)
            current_sv.append(genes)
            current_sv.append(pli_scores)
            current_sv.append(pli_count)
            current_sv.append(rvis_scores)
            current_sv.append(rvis_count)
            current_sv.append(total_num_HPO_terms)
            
        #For now, chromatin states are only checked at the exact breakpoint locations to check which functional genomic elements are disrupted. All SV types can therefore be done using the same code
        result_bp1 = index.query(current_sv[0], current_sv[1], current_sv[1]) 
        #There are three SVs that do not have a chromatin state at one of the breakpoints which screws up the np array. For these, just append 0 as a state for now. When categorical features are allowed this can become none
        if result_bp1.n_hits(2) == 0:
                current_sv.append(0)
        else:
            for hit in result_bp1[2]:
                #current_sv.append(hit.split()[3]) #If you want to append to full chromatin state name instead of only the identifier number
                state = hit.split()[3]
                current_sv.append(int(state.split("_")[0]))
        
        result_bp2 = index.query(current_sv[2], current_sv[3], current_sv[3])
        if result_bp2.n_hits(2) == 0:
                current_sv.append(0)
        else:    
            for hit in result_bp2[2]:
                #current_sv.append(hit.split()[3])
                state = hit.split()[3]
                current_sv.append(int(state.split("_")[0]))
                
        #After the SV is completely annotated, append it to a list which will contain all annotated SVs
        annotated_SVs.append(current_sv)
    
    #Convert this list to a numpy array for easier handling
    annotated_SVs = np.array(annotated_SVs, dtype="object")
    
    #Print the time needed for the annotation
    print("Compute time: %s seconds" % (time.time() - start_time))
    return annotated_SVs

tn = read_sv_data("data/TN_127.txt", 0)
tp = read_sv_data("data/TP_127.txt", 1)

annotated_tn = annotate(tn, index)
annotated_tp = annotate(tp, index)

#Quick check to see how these annotations separate the negative SV from the positive SVs
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000)
tn_tp = np.concatenate((annotated_tn, annotated_tp), axis = 0)
tn_tp_features = tn_tp[:,[7,8,11,13,14,15,16]]
#Or with the length of the SVs also as feature (much better separation):
#tn_tp_features = tn_tp[:,[6,7,8,11,13,14,15,16]]
df_tsne_tn_tp = tsne.fit_transform(tn_tp_features)

plt.scatter(df_tsne_tn_tp[:,0], df_tsne_tn_tp[:,1], c=tn_tp[:,5])
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.title("TSNE plot for annotated SVs")
plt.show()
