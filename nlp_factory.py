"The NLP Factory Class"
from bert_model import BertClass
from roberta_model import RobertaClass

class NLPFactory:  # pylint: disable=too-few-public-methods
    "The NLP Factory Class"
    @staticmethod
    def get(data_df, freeze_percent, model_name="bert-base-uncased", tokenizer="bert-base-uncased"):
        "A method to get respective model and tokenizer"
        if model_name == 'bert-base-uncased':
            #Tune BERT transformer
            return BertClass().train_model(model_name, tokenizer, data_df, freeze_percent)

        if model_name == 'roberta-base-uncased':
            return RobertaClass().train_model(model_name, tokenizer, data_df, freeze_percent)

        return None
