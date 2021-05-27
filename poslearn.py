import torch
import numpy as np
import sys
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score, classification_report
import time

start_time=time.time()

file_path = sys.argv[1]
dev_file_path = sys.argv[2]


#Hyperparameters
MAXIMUM_LENGTH_OF_SENTENCE = 100
NUMBER_OF_WORDS = 1002
EMBEDDING_SIZE = 100
HIDDEN_DIMENSION = 100
INPUT_DIM = 100
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 0.001

"""
Bidirectional LSTM MODEL
"""

class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim,input_dim, vocab_size, tagset_size):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

		# The linear layer that maps from hidden state space to tag space
		self.linear = nn.Linear(2 * hidden_dim, tagset_size)

	def forward(self, sentence,length):
		# print("Input tokens size is", sentence.size())
		
		embeds = self.word_embeddings(sentence)
		# print("Embeddings output size is" ,embeds.size())
		
		packed_embeddings = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted = False)
		# print("Packed Padded Sequence size is" , packed_embeddings.data.shape)
		
		packed_lstm_out,_ = self.lstm(packed_embeddings)#self.lstm(embeds.view(len(sentence), 1, -1))
		# print("LSTM output size is" ,packed_lstm_out.data.shape)
		

		lstm_output_padded, output_lengths = pad_packed_sequence(packed_lstm_out, batch_first=True,total_length = MAXIMUM_LENGTH_OF_SENTENCE)
		# print("Pad Packed Sequence size is ",lstm_output_padded.data.shape)
		
		
		tag_scores = self.linear(lstm_output_padded)
		# print("Linear Layer output size is" ,tag_scores.size())

		# tag_scores = F.log_softmax(tag_space, dim=1)
		# print("Softmax output size is" ,tag_space.size())


		return tag_scores





class DataGenerator():
	'''
	Takes as input the file path and generates the tag dictionary and token dictionary
	'''
	def generate_token_tag_dictionary(file):
				
		#Dictionaries to store token and tag frequencies and indexes
		tag_frequency = {}
		token_frequency = {}
		token_index = {}
		tag_index = {}

		#Reading training file  and creating term and tag frequency dictionary	
		with open(file, "r") as trainingData:
			lines = trainingData.readlines()

		for line in lines:
			line = line.lower()
			line = line.replace("\n","")
			terms = line.split(" ")
			
			for term in terms:
				items = term.split('/')
				# print(items)
				if items[0] not in token_frequency.keys():
					token_frequency[items[0]] = 1
				else:
					token_frequency[items[0]] += 1

				if items[1] not in tag_frequency.keys():
					tag_frequency[items[1]] = 1
				else:
					tag_frequency[items[1]] += 1
		
		#Sorting the token and tag dictionary by frequency and creating a index dictionary of size 1002(+ 2 for pad and unknown tokens)	
		sorted_term_dict = {k: v for k, v in sorted(token_frequency.items(), key=lambda item: -item[1])}
		sorted_tag_dict = {k: v for k, v in sorted(tag_frequency.items(), key=lambda item: -item[1])}

		print(sorted_term_dict)
		print(sorted_tag_dict)
		exit()
		
		#Appending Unknown and pad tokens to both dictionaries
		token_index['PAD']= 0
		token_index['UNKNOWN'] = 1

		tag_index['PAD']= 0
		tag_index['UNKNOWN'] = 1

		#appending frequent tokens to both the indices dictionaries
		for k, v in sorted_term_dict.items():
			if len(token_index) >= 1002:
				break;
			token_index[k] = len(token_index)
			
		for k, v in sorted_tag_dict.items():
			if len(tag_index) >= 1002:
				break;
			tag_index[k] = len(tag_index)
		

		return token_index, tag_index


	'''
	Creates the input tag tensor and token and length tensors using the two index dictioaries.
	'''
	def create_token_tag_tensors(file, token_index,tag_index):

		token_tensor = []
		tag_tensor = []
		length_tensor=[]

		with open(file, "r") as trainingData:
			lines = trainingData.readlines()
		i=0
		for line in lines:
			line = line.lower()
			line = line.replace("\n","")
			terms = line.split(" ")

			tokens = np.zeros(MAXIMUM_LENGTH_OF_SENTENCE)
			tags = np.zeros(MAXIMUM_LENGTH_OF_SENTENCE)

			min_length = min(len(terms), MAXIMUM_LENGTH_OF_SENTENCE) 
			length_tensor.append(min_length)

			j = 0
			for term in terms:
				if j>= min_length:
					break

				items = term.split('/')
				if items[0] in token_index.keys():
					tokens[j] = token_index[items[0]]
				else:
					tokens[j] = token_index['UNKNOWN']

				if items[1] in tag_index.keys():
					tags[j] = tag_index[items[1]]
				else:
					tags[j] = tag_index['UNKNOWN']
				j += 1
			token_tensor.append(tokens)
			tag_tensor.append(tags)

		token_tensor=torch.tensor(token_tensor)
		token_tensor = token_tensor.to(torch.int64)

		tag_tensor=torch.tensor(tag_tensor)
		tag_tensor = tag_tensor.to(torch.int64)

		length_tensor=torch.tensor(length_tensor)
		length_tensor = length_tensor.to(torch.int64)

		return token_tensor, tag_tensor, length_tensor



if __name__ == "__main__":

	#Creating the dictionaries and tensors for input data
	train_token_dictionary, train_tag_dictionary = DataGenerator.generate_token_tag_dictionary(file_path)
	train_token_tensor, train_tag_tensor, train_length_tensor = DataGenerator.create_token_tag_tensors(file_path, train_token_dictionary,train_tag_dictionary)

	#Reverse dictionay
	# reverse_train_token_dictionary = {v: k for k, v in train_token_dictionary.items()}	

	token_vocab_size = len(train_token_dictionary)
	tag_vocab_size = len(train_tag_dictionary)

	#Saving the training tag and token dictionariees
	with open('token_dict.json', 'w') as f:
		json.dump(train_token_dictionary, f)

	with open('tag_dict.json', 'w') as f:
		json.dump(train_tag_dictionary, f)

	
	#Preparing train data and dataloader
	train_data = torch.utils.data.TensorDataset(train_token_tensor,train_tag_tensor,train_length_tensor)
	train_data_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,sampler=train_data_sampler,shuffle = False)


	#Creating the tensors for dev data
	deb_token_tensor, dev_tag_tensor, dev_length_tensor = DataGenerator.create_token_tag_tensors(dev_file_path, train_token_dictionary,train_tag_dictionary)

	#Preparing dev data and dataloader
	dev_data = torch.utils.data.TensorDataset(deb_token_tensor,dev_tag_tensor,dev_length_tensor)
	dev_data_sampler= SequentialSampler(dev_data)
	dev_dataloader = DataLoader(dev_data, batch_size = BATCH_SIZE, sampler=dev_data_sampler, shuffle = False)

	
	#Initializing the lstm model
	model = LSTMTagger(EMBEDDING_SIZE, HIDDEN_DIMENSION, INPUT_DIM, token_vocab_size, tag_vocab_size)
	loss_function = nn.CrossEntropyLoss(ignore_index = 0)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


	#Training
	max_macro_f1 = 0
	for epoch in range(EPOCHS):  
		
		epoc_loss = 0
		#Training on the train data
		for tokens, tags, length in train_dataloader:
					
			model.zero_grad()
			
			# Run our forward pass.
			tag_scores = model(tokens,length)			
			
			# Compute the loss, gradients, and update the parameters by calling optimizer.step()	
			pred = tag_scores.view(-1,tag_vocab_size)		
			tags = tags.view(-1)
			
			loss = loss_function(pred, tags)
			epoc_loss += loss.item()			
			loss.backward()
			optimizer.step()

		epoc_loss = epoc_loss /len(train_dataloader)

		#Evaluating on dev data
		all_act = [] 
		all_pred = []

		
		with torch.no_grad():
			model.eval()
			for dev_tokens, dev_tags, dev_length in dev_dataloader:
				
				# Run our forward pass.
				dev_tag_scores = model(dev_tokens, dev_length)
				dev_tag_scores = pack_padded_sequence(dev_tag_scores, dev_length, batch_first=True, enforce_sorted = False)
				dev_tag_scores = dev_tag_scores.data
				
				dev_tokens = dev_tokens.view(-1)
				dev_tokens = dev_tokens.numpy()

				dev_tags = pack_padded_sequence(dev_tags, dev_length, batch_first=True, enforce_sorted = False)
				dev_tags =dev_tags.data
				actual_dev_tags = dev_tags.view(-1)
				actual_dev_tags = actual_dev_tags.numpy()

				predicted_dev_tags = dev_tag_scores.view(-1,tag_vocab_size)	

				values, indices = torch.max(predicted_dev_tags, dim = 1)		
				predicted_dev_tags = indices
				
				all_act.extend(actual_dev_tags)	
				all_pred.extend(predicted_dev_tags)
			

			#Removing padding and unknown tags for evaluation 
			all_act = [all_act[i] for i in range(len(all_act)) if (all_pred[i] != 0 and all_pred[i] != 1 )]				
			all_pred = [all_pred[i] for i in range(len(all_pred)) if (all_pred[i] != 0 and all_pred[i] != 1 )]       
			
			score = f1_score(all_act, all_pred, average = 'macro')
			# acc = accuracy_score(all_act, all_pred)
			# rep = classification_report(all_act, all_pred,zero_division= 0)

			print("Epoch: %d :: Training Loss is : %f ,  F1 score is : %f"% (epoch,epoc_loss,score))
				

			PATH = "lstm_model.pt"	
			if  score > max_macro_f1:
				max_macro_f1 = max(score, max_macro_f1)
				# best_rep =rep
				torch.save(model.state_dict(), PATH)
	print("Best F1- Score is %f"% (max_macro_f1))
	# print(best_rep)

	end_time=time.time()
	print("Time Taken %f"% (end_time-start_time))