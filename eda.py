" EDA funtions for common use"

class EDA:
    "EDA class implimentation"
    @staticmethod
    def is_balanced(data_df):
        "To check the data is balance or not"
        print(data_df['overall'].value_counts(normalize = True))

    @staticmethod
    def get_cleandf(data_df):
        "Filter to clean the dataset"
        data_df = data_df[data_df['reviewText'].notna()]
        data_df.overall = data_df.overall.astype(int)
        return data_df

    @staticmethod
    def split_check(train_text):
        "split check for the sentences"
        seq_len = [len(i.split()) for i in train_text]
        return seq_len
