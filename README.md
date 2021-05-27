# Part Of Speech Tagging
Part of Speech Tagging using LSTM

This is done as part of course work of CSCI-544 Natural Language Processing at University of Southern California .


poslearn.py learns a bidirectional LSTM model from labeled data, postag.py uses this model to tag new data.

poslearn.py is invoked in the following way:
>python poslearn.py train_data_path dev_data_path


It outputs a model file.  The train_data_path and dev_data_path  include the train data and dev data files respectively (not just the folders where the files are located), e.g., “python poslearn.py $ASNLIB/publicdata/train.txt $ASNLIB/publicdata/dev.txt”. The reason that poslearn.py has the dev data as input is because we can use the dev data to evaluate the model during training. After each training epoch I  save a new model only if it performs on the dev data better than the previously saved model. Evaluation is be based on macro-F1 score, which is the average of the F1 scores for each POS tag that appears in the training data. 


postag.py will be invoked in the following way:
>python postag.py validation_data_path tagged_output_file


e.g., “python postag.py ./validation.txt tagged_data.txt”. Here “validation.txt” is a file with tokens but without any POS tags. For example, if the validation file contains the sentence “I guess ,” then the tagged output file would contain something like “i/prp guess/vbp ,/,” or “I/PRP guess/VBP ./.”. The validation file will contain tokenized sentences. So it won’t have sentences like “I guess, this is good.” but “I guess , this is good .” instead.


Working

After reading the training data, two dictionaries are created, one for tokens and one for POS tags. The dictionary of tokens only uses the 1000 most frequent tokens in the training data and also has extra entries for padding tokens and unknown tokens. The value for the padding entry could be 0 and the value for an unknown token could be 1. Likewise for POS tags. It is possible that a POS tag that appears in the dev data (or test data) is not included in the training data.  When the dev or test data are read each token is assigned the corresponding value from the tokens dictionary extracted from the training data unless it is an unknown token. Likewise for POS tags.

The next step is to create matrices for the tokens and the POS tags with dimensions number_of_sentences x maximum_length_of_sentence. As the maximum length of a sentence is kept 100 (you can try with length of longest sentence). An additional matrix is created storing the length of each sentence. This is done for both the training and the dev data. Then tensors for the tokens, tags, and lengths matrices are created. I use DataLoader for loading the samples.
Using 1-hot encoding did not result in good performance. So I uses embeddings.
