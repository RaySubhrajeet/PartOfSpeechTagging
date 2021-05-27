import torch
import numpy as np
import json
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score
from poslearn import LSTMTagger 
import time

start_time = time.time()

file_path = sys.argv[1]
output_path = sys.argv[2]

MAXIMUM_LENGTH_OF_SENTENCE=100

def create_test_tensors(file, token_index):
	token_tensor = []
	length_tensor=[]

	with open(file, "r") as testData:
		lines = testData.readlines()
	i=0
	for line in lines:
		line = line.lower()
		line = line.replace("\n","")
		terms = line.split(" ")

		tokens = np.zeros(100)

		min_length = min(len(terms), 100) 
		length_tensor.append(min_length)

		j = 0
		for term in terms:
			if j>= min_length:
				break			
			if term in token_index.keys():
				tokens[j] = token_index[term]
			else:
				tokens[j] = token_index['UNKNOWN']
			
			j += 1
		token_tensor.append(tokens)
		
	token_tensor=torch.tensor(token_tensor)
	token_tensor = token_tensor.to(torch.int64)

	length_tensor=torch.tensor(length_tensor)
	length_tensor = length_tensor.to(torch.int64)

	return token_tensor, length_tensor





token_dict = {}
tag_dict = {}
with open('token_dict.json') as f:
	token_dict = json.load(f)

with open('tag_dict.json') as f:
	tag_dict = json.load(f)
# model = torch.load(PATH)
# model.eval()
PATH = "lstm_model.pt"
model = LSTMTagger(100, 100, 100, len(token_dict), len(tag_dict))
model.load_state_dict(torch.load(PATH))
model.eval()


#Creating the dictionaries and tensors for test data

test_token_tensor, test_length_tensor = create_test_tensors(file_path, token_dict)
reverse_tag_dictionary = {v: k for k, v in tag_dict.items()}

test_data = torch.utils.data.TensorDataset(test_token_tensor,test_length_tensor)
test_data_sampler= SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=100,sampler=test_data_sampler, shuffle = False)

prediction = []
with torch.no_grad():
	for test_tokens, test_length in test_dataloader:		
		test_tag_scores = model(test_tokens, test_length)
		
		predicted_test_tags = []
		for score in test_tag_scores:	
			values, indices = torch.max(score, dim = 1)
			indices =  [indices[i].item() for i in range(len(indices)) if (indices[i] != 0)] 
			predicted_test_tags.append(indices)			
		
		prediction.extend(predicted_test_tags)

	
with open(file_path, "r") as testData:
		lines = testData.readlines()

f = open(output_path, "w")
i = 0
for line in lines:
	line = line.lower()
	line = line.replace("\n","")
	terms = line.split(" ")
	
	j = 0
	for term in terms:
		if j== 0:

			f.write(term +"/"+ reverse_tag_dictionary[prediction[i][j]])
		else:
			f.write(" " + term +"/"+ reverse_tag_dictionary[prediction[i][j]])
		j += 1
	i += 1

	f.write("\n")
f.close()
end_time = time.time()
print("Total time: %f"% (end_time-start_time))


