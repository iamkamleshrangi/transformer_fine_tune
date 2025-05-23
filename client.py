"Main driver script for data handling and training"
import argparse
import pandas as pd
from nlp_factory import NLPFactory
from config.config_handler import handler

def client():
    "Driver function"
    #Argument parser to handle defualt or custom tuning parameters
    arguments = argparse.ArgumentParser()
    arguments.add_argument("-m", "--model", required=False,\
        default = handler('model','name'), help="Model name to be used")
    arguments.add_argument("-t", "--tokenizer", required=False,\
        default = handler('model','tokenizer'), help="Tokenizer name to be used")
    arguments.add_argument("-d", "--dataset", default= handler('settings','dataset'),\
        required=False, help="Training dataset to do finetuning.")
    arguments.add_argument("-f", "--freeze", default= handler('settings','elsa_percent'),\
        help="Layer freezing percentage")

    args = vars(arguments.parse_args())

    #Load default or input dataset in a dataframe
    with open(args['dataset'], encoding='utf-8-sig') as f_input:
        data_df = pd.read_json(f_input, lines=True)
        print(data_df.shape)
    #NLP Factory to tune input transfore based on ELSA research
    NLPFactory().get(data_df, args['freeze'], args['model'], args['tokenizer'])

client()
