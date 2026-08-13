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
    
def process_iwslt_input():
    df_train = pd.read_json(config.DATASET_DIR / "iwslt_train.jsonl", lines=True, orient='records')
    df_test = pd.read_json(config.DATASET_DIR / "iwslt_test.jsonl", lines=True, orient='records')

    zh_tokenizer = ZHTokenizer.from_vocab(config.DATASET_DIR / "iwslt_train_zh_vocab.txt")
    df_train["zh"] = df_train["zh"].apply(lambda x: zh_tokenizer.encode(x))
    def make_windows(zhlist, window_size=config.INPUY_WINDOW_SIZE):
        # 按窗口滑动：窗口内 W 个 token 作 input，窗口后一位 token 作 target
        W = window_size
        windows = []
        for sentence in zhlist:
            # len(sentence) - W 保证最后窗口后仍有 target 可取
            for i in range(len(sentence) - W):
                windows.append({
                    "input": sentence[i:i + W],
                    "target": sentence[i + W],
                })
        return windows

    result = make_windows(df_train["zh"].tolist())
    print(f"共生成 {len(result)} 条样本，示例：{result[0] if result else '无'}")
    pd.DataFrame(result).to_json(config.DATASET_DIR / "iwslt_train_tokenized_input.jsonl",
                                 orient="records", lines=True, force_ascii=False)

    df_test["zh"] = df_test["zh"].apply(lambda x: zh_tokenizer.encode(x))
    result_test = make_windows(df_test["zh"].tolist())
    pd.DataFrame(result_test).to_json(config.DATASET_DIR / "iwslt_test_tokenized_input.jsonl",
                                      orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    process_iwslt_input()
