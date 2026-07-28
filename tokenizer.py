from abc import ABC, abstractmethod
import jieba

class Tokenizer(ABC):
    oov = "<unk>"
    pad = "<pad>"
    sos = "<sos>"
    eos = "<eos>"
    def __init__(self, vocab_list):
        self.word2id = {word: i for i, word in enumerate(vocab_list)}
        self.id2word = {i: word for i, word in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)
        self.oov_id = self.word2id[self.oov]
        self.pad_id = self.word2id[self.pad]
        self.sos_id = self.word2id[self.sos]
        self.eos_id = self.word2id[self.eos]

    @classmethod
    @abstractmethod
    def tokenize(cls, sentence):
        """子类必须实现此方法，定义分词逻辑"""
        pass

    def encode(self, sentence):
        return [self.word2id.get(token, self.oov_id) for token in self.tokenize(sentence)]
    
    @classmethod
    def build_vocab(cls, sentences, path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad, cls.oov, cls.sos, cls.eos] + list(vocab_set)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))
    
    @classmethod
    def from_vocab(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab_list = f.read().splitlines()
        return cls(vocab_list)
        

class ENTokenizer(Tokenizer):
    @classmethod
    def tokenize(cls, sentence):
        from nltk import word_tokenize
        return word_tokenize(sentence)

class ZHTokenizer(Tokenizer):
    @classmethod
    def tokenize(cls, sentence):
        return jieba.lcut(sentence)

class JiebaTokenizer(Tokenizer):
    @classmethod
    def tokenize(cls, sentence):
        return jieba.lcut(sentence)
