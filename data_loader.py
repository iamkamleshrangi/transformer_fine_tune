"Data sampling and tensor conversions module"
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def split_data(text, labels, split_size):
    "Splits data into train and test with input split size"
    a_text, b_text, a_labels, b_labels = \
    train_test_split(text, labels,
                    random_state=2018,
                    test_size=split_size,
                    stratify=labels)
    return (a_text, b_text, a_labels, b_labels)

def tensorize(tokens, labels):
    "Converts tokens into tensors"
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y_labels = torch.tensor(labels.tolist()) - 1
    return (seq, mask, y_labels)

class DataLoad:
    "Class to handle data sampling and tensor conversions"
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = 200
        #Batch size as defined in ELSA
        self.batch_size = 16

    def tokenize(self, data):
        "Tokenize and encode sequences"
        tokens = self.tokenizer.batch_encode_plus(
            data.tolist(),
            max_length = self.max_seq_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False
        )
        return tokens


    def execute(self):
        "Driver for sampling and conversions"
        #Split Training data into train and validation sets to be used for tuning
        train_text, val_text, train_labels, val_labels \
        = split_data(self.dataframe['reviewText'], self.dataframe['overall'], 0.3)
        #Convert data into tensors for pytorch
        train_seq, train_mask, train_y = \
        tensorize(self.tokenize(train_text), train_labels)
        val_seq, val_mask, val_y = \
        tensorize(self.tokenize(val_text), val_labels)

        #Convert Tensors into Dataloader for iteration ease while tuning
        train_data = TensorDataset(train_seq, train_mask, train_y)
        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)

        # dataLoader for train set
        train_data = DataLoader(train_data, sampler=train_sampler,\
        batch_size = self.batch_size)

        # wrap tensors
        val_data = TensorDataset(val_seq, val_mask, val_y)

        # sampler for sampling the data during training
        val_sampler = SequentialSampler(val_data)

        # dataLoader for validation set
        val_data = DataLoader(val_data, sampler = val_sampler, batch_size = self.batch_size)

        return(train_data, train_labels, val_data)
