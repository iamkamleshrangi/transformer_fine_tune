"Bert Model load and process"
from transformers import AutoModel, BertTokenizerFast
from interface_model import IModel
from eda import EDA
from data_loader import DataLoad
from network_architecture import ModelArch
from trainer import Trainer

class BertClass(IModel):
    "Bert Model Interface"

    @staticmethod
    def load(model_name, tokenizer_name):
        "Load and return BERT base uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        return model, tokenizer

    def train_model(self, model_name, tokenizer_name, data_df, freeze_percent):
        "Tunes BERT Language Model"
        #Load respective Model and Tokenizer
        transfomer_model, tokenizer = self.load(model_name, tokenizer_name)
        #Basic EDA and preprocess data
        data_df = EDA().get_cleandf(data_df)
        #Convert Dataset into tensors for training
        train_dataloader, train_labels, \
        val_dataloader = DataLoad(data_df, tokenizer).execute()
        #Pass BERT for layer freezing and setup basic network
        model = ModelArch(transfomer_model, freeze_percent)
        #Execute tuning and validation
        Trainer(model, train_dataloader, train_labels, val_dataloader).execute()
