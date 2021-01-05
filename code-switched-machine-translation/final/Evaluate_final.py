import nltk
from jiwer import wer
import os
from os import listdir
from os.path import isfile, join


def load_data(path):
	with open(path, 'r') as file:
			contents = file.read().splitlines()

	return contents

def calculate_bleu_score(ground_truth_list, prediction_list, weights=(0.8,0.2,0,0)):
	# bleu: higher the better
	# weights is a tuple of weightage to (unigram, bigram, trigram, 4-gram)
	if len(ground_truth_list) == len(prediction_list):
		bleu_sum = 0
		x = nltk.translate.bleu_score.SmoothingFunction()
		for ii in range(len(prediction_list)):
			try:
					hypothesis = prediction_list[ii].lower().split()
					reference = ground_truth_list[ii].lower().split()
					
					BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis,smoothing_function=x.method1, weights=weights)
					bleu_sum += BLEUscore
			except Exception as e:
					#print("EXCEPTION: ", e)
					pass

		return str(bleu_sum/len(prediction_list))
	else:
		print("Error! ground truth and prediction list sizes dont match")
		return str(0)


def calculate_wer(ground_truth_list, prediction_list):
	# wer: lower the better
	if len(ground_truth_list) == len(prediction_list):
		total_wer = 0.0
		for ii in range(len(prediction_list)):
				try:
						hypothesis = prediction_list[ii].lower().split()
						reference = ground_truth_list[ii].lower().split()

						err = wer(reference, hypothesis, standardize=True)
						total_wer += err
				except Exception as e:
					#print("EXCEPTION: ", e)
					pass

		return str(total_wer/len(prediction_list))
	else:
		print("Error! ground truth and prediction list sizes dont match")
		return str(1.0)

gold_data = load_data('test.txt')

# calculate bleu 1, bleu 2, bleu 3, bleu 4, wer on transformer NMT
print("Transformer NMT:")
predictions1 = load_data('predictions1_improvement1.txt')
print("BLEU-1: " + calculate_bleu_score(gold_data, predictions1, (1, 0, 0, 0)))
print("BLEU-2: " + calculate_bleu_score(gold_data, predictions1, (0, 1, 0, 0)))
print("BLEU-3: " + calculate_bleu_score(gold_data, predictions1, (0, 0, 1, 0)))
print("BLEU-4: " + calculate_bleu_score(gold_data, predictions1, (0, 0, 0, 1)))
print("Avg BLEU: " + calculate_bleu_score(gold_data, predictions1, (0.25, 0.25, 0.25, 0.25)))
print("WER: " + calculate_wer(gold_data, predictions1) + "\n")


# calculate bleu 1, bleu 2, bleu 3, bleu 4, wer on transformer NMT with finetuning
print("Transformer NMT with finetuning:")
predictions2 = load_data('predictions2_improvement2.txt')
print("BLEU-1: " + calculate_bleu_score(gold_data, predictions2, (1, 0, 0, 0)))
print("BLEU-2: " + calculate_bleu_score(gold_data, predictions2, (0, 1, 0, 0)))
print("BLEU-3: " + calculate_bleu_score(gold_data, predictions2, (0, 0, 1, 0)))
print("BLEU-4: " + calculate_bleu_score(gold_data, predictions2, (0, 0, 0, 1)))
print("Avg BLEU: " + calculate_bleu_score(gold_data, predictions2, (0.25, 0.25, 0.25, 0.25)))
print("WER: " + calculate_wer(gold_data, predictions2) + "\n")


# calculate bleu 1, bleu 2, bleu 3, bleu 4, wer on transformer NMT initialized with embeddings and finetuned
print("Transformer NMT with finetuning and embeddings:")
predictions3 = load_data('predictions3_improvement3.txt')
print("BLEU-1: " + calculate_bleu_score(gold_data, predictions3, (1, 0, 0, 0)))
print("BLEU-2: " + calculate_bleu_score(gold_data, predictions3, (0, 1, 0, 0)))
print("BLEU-3: " + calculate_bleu_score(gold_data, predictions3, (0, 0, 1, 0)))
print("BLEU-4: " + calculate_bleu_score(gold_data, predictions3, (0, 0, 0, 1)))
print("Avg BLEU: " + calculate_bleu_score(gold_data, predictions3, (0.25, 0.25, 0.25, 0.25)))
print("WER: " + calculate_wer(gold_data, predictions3))
