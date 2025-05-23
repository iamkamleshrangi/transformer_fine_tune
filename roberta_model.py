"RoBerta Model load and process"
from transformers import RobertaModel, RobertaTokenizer
from interface_model import IModel
from eda import EDA
from data_loader import DataLoad
from network_architecture import ModelArch
from trainer import Trainer

class RobertaClass(IModel):
    "RoBerta Model Interface"

    @staticmethod
    def load(model_name, tokenizer_name):
        "Load and return RoBerta base uncased"
        model = RobertaModel.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        return model, tokenizer

    def train_model(self, model_name, tokenizer, data_df, freeze_percent):
        "Train model RoBerta"
        #Load respective Model and Tokenizer
        transfomer_model, tokenizer = self.load(model_name, tokenizer)
        #Basic EDA and preprocess data
        data_df = EDA().get_cleandf(data_df)
        #Convert Dataset into tensors for training
        train_dataloader, train_labels, val_dataloader\
        = DataLoad(data_df, tokenizer).execute()
        #Pass RoBerta for layer freezing and setup basic network
        model = ModelArch(transfomer_model, freeze_percent)
        #Execute tuning and validation
        Trainer(model, train_dataloader, train_labels, val_dataloader).execute()
