import pandas as pd
import config
from sklearn.model_selection import train_test_split
import jieba
from tokenizer import ZHTokenizer

def build_dataset(sentences, tokenizer):
    # indexed = [[word2id.get(token, 0) for token in Tokenizer.tokenize(sentence)] for sentence in sentences]
    indexed = [tokenizer.encode(sentence) for sentence in sentences]
    dataset = []
    for data in indexed:
        for i in range(len(data) - config.WINDOW_SIZE):
            inp = data[i:i + config.WINDOW_SIZE]
            target = data[i + config.WINDOW_SIZE]
            dataset.append({"input": inp, "target": target})
    return dataset


def process():
    raw_json = pd.read_json(config.DATASET_DIR / "raw.jsonl", lines=True, orient='records')
    # print(raw_json.head().to_dict())
    dialog = raw_json["dialog"]
    sentences = []
    for d in dialog:
        for s in d:
            sentences.append(s.split("：")[1].strip())
    train_data, test_data = train_test_split(sentences, test_size=0.2, random_state=42)

    vocab_path = config.DATASET_DIR / "vocab.txt"
    # vocab_set = set()
    # for sentence in train_data:
    #     vocab_set.update(jieba.cut(sentence))
    
    # vocab_list = ["<unk>"] + list(vocab_set)
    # word2id = {word: i for i, word in enumerate(vocab_list)}
    # id2word = {i: word for i, word in enumerate(vocab_list)}

    
    # with open(vocab_path, "w", encoding="utf-8") as f:
    #     for token in vocab_list:
    #         f.write(token + "\n")
    ZHTokenizer.build_vocab(train_data, vocab_path)
    tokenizer = ZHTokenizer.from_vocab(vocab_path)

    # print(sentences[0:10])
    train_dataset = build_dataset(train_data, tokenizer)
    test_dataset = build_dataset(test_data, tokenizer)

    train_dataset_path = config.DATASET_DIR / "train_data_set.jsonl"
    pd.DataFrame(train_dataset).to_json(train_dataset_path, orient="records", lines=True, force_ascii=False)

    test_dataset_path = config.DATASET_DIR / "test_data_set.jsonl"
    pd.DataFrame(test_dataset).to_json(test_dataset_path, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    process()