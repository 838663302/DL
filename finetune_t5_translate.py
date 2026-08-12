"""
T5 中文翻译微调示例
用法：
   1. 先准备好 t5_train.jsonl（每行 {"source": "...", "target": "..."}）
   2. python finetune_t5_translate.py
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ---------- 数据集 ----------
class TranslationDataset(Dataset):
    """读取 jsonl，每行 {"source": "...", "target": "..."} """
    def __init__(self, file_path, tokenizer, max_source_len=128, max_target_len=128):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 输入：任务前缀 + 中文原文
        source_enc = self.tokenizer(
            item["source"],
            max_length=self.max_source_len,
            truncation=True,
            padding=False,
            return_tensors=None,   # 返回 list，由 collator 做 batch padding
        )
        # 目标：英文译文
        target_enc = self.tokenizer(
            item["target"],
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


# ---------- 主流程 ----------
def main():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_dataset = TranslationDataset("t5_train_raw.jsonl", tokenizer)
    # val_dataset  = TranslationDataset("t5_val_raw.jsonl",   tokenizer)   # 如果有验证集

    # collator 负责将不等长序列在 batch 内 padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-zh-en-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        predict_with_generate=True,          # 评估时用 generate() 输出 BLEU
        fp16=torch.cuda.is_available(),      # 有 GPU 自动开混合精度
        report_to="none",                    # 不上传 wandb
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./t5-zh-en-translator")

    # ---------- 测试 ----------
    test_text = "translate Chinese to English: 这是一封索赔提醒邮件。"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    print("翻译结果:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
