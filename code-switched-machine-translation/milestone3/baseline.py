import nltk
import ssl
import csv
from googletrans import Translator
import time
import re
from indictrans import Transliterator
from litcm import LIT
from os import listdir
from os.path import isfile, join


try:
	_create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
	pass
else:
	ssl._create_default_https_context = _create_unverified_https_context
nltk.download('words')


trn = Transliterator(source='hin', target='eng', build_lookup=True)



def preprocess_count_longest_phrase(hinglish_file_path, embedded_file_path):
	lit = LIT(labels=['hin', 'eng'], transliteration = False)
	f = open(hinglish_file_path, 'r')
	g = open(embedded_file_path, 'w+')
	for line in f:
		# add space around punctuation according to litcm rules
		reg = "(^[^a-zA-Z0-9]+|[^-'a-zA-Z0-9]+|[^a-zA-Z0-9]+$)"
		line = re.sub(reg, r' \1 ', line)

		tagged_s = (lit.identify(line))
		#print(tagged_s)
		en_c = 0
		hi_c = 0
		tokens = tagged_s.split()
		for word in tokens:
			if word[-3:] == 'Eng':
				en_c += 1
			elif word[-3:] == 'Hin':
				hi_c += 1
		if hi_c and en_c:
			if hi_c <= en_c:
				tag = 'Hin'
			else:
				tag = 'Eng'
			max_c = 0
			best_i = -1
			i = 0
			while i < len(tokens):
				if tokens[i][-3:] == tag:
					j = i + 1
					while j < len(tokens) and (tokens[j][-3:] == tag or tokens[j][-3:] == '\O'):
						j += 1
					if j - i > max_c:
						max_c = j - i
						best_i = i
					i = j
				else:
					i += 1
			g.write(tag + ' ' + str(best_i) + ' ' + str(best_i + max_c) + '\n')
			#print(tokens[best_i:best_i+max_c])
		else:
			g.write('None\n')

	f.close()
	g.close()



def replace_longest_phrases(hinglish_file_path, embedded_file_path, embedded_translated_file_path='augmented.txt'):
	
	reg = "(^[^a-zA-Z0-9]+|[^-'a-zA-Z0-9]+|[^a-zA-Z0-9]+$)"

	with open(embedded_file_path, 'r') as file:
		aug_rules = file.read().splitlines()

	with open(hinglish_file_path, 'r') as file:
		hi_contents = file.read().splitlines()

	# split punctuation according to litcm rules
	reg = "(^[^a-zA-Z0-9]+|[^-'a-zA-Z0-9]+|[^a-zA-Z0-9]+$)"
	hi_contents = [re.sub(reg, r' \1 ', line) for line in hi_contents]

	prev = -1
	new_contents = []

	for i in range(prev+1, len(aug_rules)):
		try:
			splits = aug_rules[i].split()
			# if no phrase detected here
			if len(splits) < 2:
				prev += 1
				new_contents.append(hi_contents[i])
				continue

			if splits[0] == 'Eng':
				target = 'hi'
			else:
				target = 'en'
			start_idx = int(splits[1])
			end_idx = int(splits[2])
			print(str(i) + ": ")
			print(target_lang, start_idx, end_idx)

			sentence = hi_contents[i].split()
			to_replace = sentence[start_idx:end_idx]
			# translate call
			replaced = translator.translate(to_replace, dest=target)
			print("Transforming this: " + replaced[0].text)
			eng_text = trn.transform(replaced[0].text)
			print(str(i) + ": Replacing this phrase: " + " ".join(to_replace) + " :WITH: " + eng_text)
			sentence[start_idx:end_idx] = eng_text.split(' ')
			new_contents.append(" ".join(sentence))
			prev += 1
		except Exception as e:
			print(e)
			print("ERROR!: prev = " + str(prev) + " and i = " + str(i))
			print(hi_contents[i])
			break

	with open(embedded_translated_file_path, 'w+') as f:
		for item in new_contents:
			f.write("%s\n" % item)



def load_data(path):
	with open(path, 'r') as file:
		contents = file.read().splitlines()

	return contents


def make_batch_api_calls_to_gnmts(input_list, output_file_path, batch_size=300, target_language='en'):
	# target language can be 'en' or 'hi' in our case
	prev = -1
	f = open(output_file_path, 'w+')
	for batch_no in range(prev+1,int(len(input_list)/batch_size)+1):
		try:
			translator = Translator()
			if (batch_no+1)*batch_size < len(input_list):
				batch_item = input_list[batch_no*batch_size:(batch_no+1)*batch_size]
			else:
				batch_item = input_list[batch_no*batch_size:len(input_list)+1]

			batch_res = translator.translate(batch_item, dest=target_language)
			translated_batch_text = [f.write(item.text + "\n") for item in batch_res]
			time.sleep(30)  
			print("batch ", batch_no, ": done.")
			prev += 1
		except Exception as e:
			# if error is : Expecting value: line 1 column 1 (char 0)
			# its because gnmts api call blacklist ip if continuous api calls are made
			print(e)
			print("ERROR! ", batch_no, "unsuccessful. If error is Expecting value: line 1 column 1 (char 0), its because google nmts blacklists an ip if continuous api calls are made! Sorry try again!")
			break

	f.close()
		

def main():

	en_contents = load_data('./code_mixed_dataset/english.txt')
	hien_contents = load_data('./code_mixed_dataset/hinglish.txt')

	# this creates a file containing start and end index of longest strings in embedded language
	# we will translate these strings to Matrix language using google translate
	preprocess_count_longest_phrase('./code_mixed_dataset/hinglish.txt', './embedded.txt')

	if isfile('augmented.txt'):
		augmented = load_data('augmented.txt')
	else:
		replace_longest_phrases('./code_mixed_dataset/hinglish.txt', './embedded.txt', 'augmented.txt')
		augmented = load_data('augmented.txt')


	if not isfile("gnmts_translated.txt"):
		make_batch_api_calls_to_gnmts(augmented, "gnmts_translated.txt")	


if __name__ == "__main__":
	main()
	