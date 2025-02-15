from __future__ import absolute_import
from copy import deepcopy

class TAD:
	
	"""
		Class to describe the location of a TAD and the genes and genomic elements found within this TAD. 
	"""
	
	def __init__(self, chromosome, start, end):
		
		self.chromosome = chromosome
		self.start = start
		self.end = end
		self.SVs = None
		self.elements = []
		self.elementsStr = dict()
		self.genes = []
		self.startStrength = '0'
		self.endStrength = '0'
		self.startStrengthSignal = '0'
		self.endStrengthSignal = '0'
		
	def setSVs(self, SVs): #All SVs that overlap with this TAD
		
		self.SVs = SVs
	
	def setElements(self, elements): #All regulatory elements found inside this TAD
		self.elements = elements
	
	def setElementsStr(self, elements): #str version to quickly match elements in the TAD
		for element in elements:
			elementStr = element[0] + "_" + str(element[1]) + "_" + str(element[2]) + "_" + str(element[3]) + "_" + str(element[4])
			self.elementsStr[elementStr] = 0
				
	def addElements(self, elements): #add nwe elements to the TAD
		self.elementsStr = dict()
		for element in elements:
			self.elements.append(list(element))
		
	def addGene(self, gene): #Function to add genes that are within the TAD, one-by-one
		self.genes.append(gene)
		
	def setGenes(self, genes): #add genes to the TAD
		self.genes = genes
	
	#Get all genomic elements within this TAD given a specific range within the TAD
	def getElementsByRange(self, start, end):
		
		elementsInRange = []
		for element in self.elements:
			if element[2] >= start and element[1] <= end:
				elementsInRange.append(element)

		return elementsInRange
	
	#Get all genes within this TAD given a specific range within the TAD
	def getGenesByRange(self, start, end):
		genesInRange = []
		for gene in self.genes:
			if gene.start >= start and gene.start <= end or gene.end <= end and gene.end >= start:
				genesInRange.append(gene)
		
		return genesInRange
	