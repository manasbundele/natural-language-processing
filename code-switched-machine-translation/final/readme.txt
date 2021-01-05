Team Members: Randeep Ahlawat, Siddhartha Sahai, Manas Bundele

We have 3 separate colab links containing the code and executed output and calculated metrics for the 3 improvements. 

We also include a single evaluate.py file that calculates BLEU-1,2,3,4 and WER for the translated predictions.

Run this file using:
'pip3 install jiwer nltk'
'python3 evaluate.py'


NMT model without finetuning:
https://colab.research.google.com/drive/16DCwO79D_SNPiATNmGu3orlQ9G7jB4Ch

NMT model with finetuning:
https://colab.research.google.com/drive/1NxTGw3slLY82f-4CCwBGcMsIsb8uJ6Dh

NMT model using embeddings with finetuning:
https://colab.research.google.com/drive/1utXSCTRcXTaxrTJOLPaAi0HldaWtQUxf


Since the model files are large, we have included google drive links to the generated models (corresponding to the 3 improvements):

Model 1 (transformer nmt):
https://drive.google.com/open?id=1-HKcrPLGAxkO0bqhgHi-XDhfXKZt8MuS

Model 2 (finetuned nmt):
https://drive.google.com/open?id=1-G2InCUcyx_9Z949NzQSbGuPU9SGAxzL

Model 3 (finetuned nmt with embedding):
https://drive.google.com/open?id=1-LVRDYgHKhTNOfQ2KlYwmilbZgHV7gD1


The test set for the models are the src-test-bpe.txt and tgt-test-bpe.txt files located in this folder:
https://drive.google.com/open?id=1NML1_1xzk5SMZonXgtiBwX1UVNtOG5a4

(it also requires the translations to be converted back to readable format using this command: !sed -r 's/(@@ )|(@@ ?$)//g' input_file > output_file)
