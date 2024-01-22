from cbart.models.cbartmain import *
import torch
from loadconfig import loadConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from clipscore.clipscore import * 
from bertPPL.run_lm_predict import *
from yolov5.yolov5 import getTag
correct_phrase = loadConfig('Rank')

def getRoberta():
	roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	roberta.cuda()
	roberta.eval()
	return roberta

def getContradictionScores(roberta,sentences,rov):
	'''
		Input：
			robeta: model
			sentences: retrieved sentences containing results of common sense reasoning
			rov: sentence after reversing of rank
		Output：
			scores: contradiction scores
	'''
	scores = []
	for sent in sentences:
		tokens = roberta.encode(rov, sent)
		value = roberta.predict('mnli', tokens).cpu().detach().numpy()
		value = round(value[0].tolist()[0],3)
		scores.append((value,sent.capitalize()))
	return scores

def getImageTextRelationScores(roberta,sentences,rov,imagePath):
	'''
		Input：
			robeta: model
			sentences: retrieved sentences containing results of common sense reasoning
			rov: sentence after reversing of rank
			image: image path
		Output：
			value: image-text relation score
			scores_imageTextRelation: each sentence and its image-text relevance score
	'''
	scores_imageTextRelation = []
	for sent in sentences: 
		value = clipscore(sent, imagePath)
		scores_imageTextRelation.append((value,sent.capitalize()))
	return scores_imageTextRelation

def getGrammaScores(sentences): 
	'''
		Input：
			robeta: model
			sentences: retrieved sentences containing results of common sense reasoning
			rov: sentence after reversing of rank
		Output：
			scores: gramma scores
	'''
	scores = []
	for sent in sentences: 
		value = ppl(sent)
		scores.append((value,sent.capitalize()))
	return scores

def getOverallScores(scores_contradictory, scores_imageTextRelation, scores_gramma):
	scores = []
	for i in range(len(scores_contradictory)):
		scores.append((-1*scores_contradictory[i][0]*scores_imageTextRelation[i][0]*scores_gramma[i][0],scores_contradictory[i][1].capitalize()))
	print("getOverallScores：",scores)
	return scores

def getOverallScores_woI(scores_contradictory, scores_imageTextRelation, scores_gramma):
	scores = []
	for i in range(len(scores_contradictory)):
		scores.append((-1*scores_contradictory[i][0]*scores_gramma[i][0],scores_contradictory[i][1].capitalize()))
	print("getOverallScores：",scores)
	return scores

def rank_sentences_based_on_contradiction(roberta, sentences, rov, imagePath):
	'''
		Input：
			robeta: model
			sentences: retrieved sentences containing results of common sense reasoning
			rov: sentence after reversing of rank
		Output：
			scores[0]:
	'''
	scores_contradictory = getContradictionScores(roberta,sentences,rov) # get score and sentence of contradiction score
	scores_imageTextRelation = getImageTextRelationScores(roberta,sentences,rov,imagePath) # get image text relation scores

	scores_gramma = getGrammaScores(sentences) # PPL 
	scores = getOverallScores(scores_contradictory,scores_imageTextRelation,scores_gramma) 
	# scores = getOverallScores_woI(scores_contradictory,scores_imageTextRelation,scores_gramma) # woI
	# scores = getContradictionScores(roberta,sentences,rov) 

	scores.sort(key = lambda x: (x[0],-len(x[1].split())),reverse=True) # Sort the resulting sentences and scores
	print("scores[0]:", scores[0]) # scores[0]: (-0.49, 'The person who died unbrella , got a wet suit .')
	return scores[0]


def rankContext(roberta, rov, commonsense, imagePath, extra=''):
	tag = getTag(imagePath)
	sentences = cbart(commonsense, tag)

    # select most contradictory text
	mostcontradictory = rank_sentences_based_on_contradiction(roberta, sentences, rov, imagePath) 
	x = mostcontradictory[1].capitalize() # get most contradictory text
	x = x.replace(' i ',' I ')
	return x




	
