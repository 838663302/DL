import pandas as pd
import config
from tokenizer import ZHTokenizer, ENTokenizer

def process_iwslt():
    df_train = pd.read_json(config.DATASET_DIR / "iwslt_train.jsonl", lines=True, orient='records')
    df_test = pd.read_json(config.DATASET_DIR / "iwslt_test.jsonl", lines=True, orient='records')
    ENTokenizer.build_vocab(df_train["en"], config.DATASET_DIR / "iwslt_train_en_vocab.txt")
    ZHTokenizer.build_vocab(df_train["zh"], config.DATASET_DIR / "iwslt_train_zh_vocab.txt")

    zh_tokenizer = ZHTokenizer.from_vocab(config.DATASET_DIR / "iwslt_train_zh_vocab.txt")
    en_tokenizer = ENTokenizer.from_vocab(config.DATASET_DIR / "iwslt_train_en_vocab.txt")
    df_train["zh"] = df_train["zh"].apply(lambda x: zh_tokenizer.encode(x))
    df_train["en"] = df_train["en"].apply(lambda x: en_tokenizer.encode(x, sos_eos=True))
    df_test["zh"] = df_test["zh"].apply(lambda x: zh_tokenizer.encode(x))
    df_test["en"] = df_test["en"].apply(lambda x: en_tokenizer.encode(x, sos_eos=True))

    df_train.to_json(config.DATASET_DIR / "iwslt_train_tokenized.jsonl", orient="records", lines=True, force_ascii=False)
    df_test.to_json(config.DATASET_DIR / "iwslt_test_tokenized.jsonl", orient="records", lines=True, force_ascii=False)
    

if __name__ == "__main__":
    process_iwslt()
