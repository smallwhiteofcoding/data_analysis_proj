import csv  # csv读取
import re  # 正则表达
import torch  # 深度学习库
import random
import numpy as np
import pandas as pd
from rouge import Rouge  # 用于评估摘要质量的库
from tqdm import tqdm  # 用于生成进度条
from transformers import pipeline  # 用于生成摘要的库
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments  # 用于分词，和模型
from datasets import Dataset

# 预处理训练集
train_path = '../T5_datasets/train_dataset.csv'  # 自定义训练集路径
train = pd.read_csv(train_path, sep='\t', names=["Index", "Text", "Abstract"])
train_dataset = Dataset.from_pandas(train)
train_dataset = train_dataset.train_test_split(100, seed=42)

# 1.模型加载
torch=torch.device("gpu")
model_name = "../T5"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 2.数据处理
def process_func(examples):
    inputs = tokenizer(examples['Text'], max_length=512, truncation=True)
    # content=[]
    # for e in examples['Abstract']:
    #     content.append(str(e))
    # labels = tokenizer(content, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Abstract"], max_length=512, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenizer_ds = train_dataset.map(process_func, batched=True)
print(tokenizer_ds)

#3.创建模型
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
rouge = Rouge()
#4.创建评估函数
def compute_metric(evalPred):

    predictions,labels=evalPred
    output = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100,labels,tokenizer.pad_token_id)
    d_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_score = rouge.get_scores(output, d_labels,avg=True)# 使用 Rouge 库计算 rouge_L 分数
    return {
        'rouge-1': rouge_score["rouge-1"]['f'],
        'rouge-2': rouge_score["rouge-2"]['f'],
        'rouge-l': rouge_score["rouge-l"]['f'],
    }




#5.配置训练参数,微调
training_args = Seq2SeqTrainingArguments(
    output_dir='../T5',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    logging_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="rouge-l",
    predict_with_generate=True,
    fp16=True,
)


#6.创建训练模型
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenizer_ds["train"],
    eval_dataset=tokenizer_ds["test"],
    compute_metrics=compute_metric,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)

trainer.train()


#6.预测
summarizier = pipeline('summarization', model=model, tokenizer=tokenizer)
for i in range(5):
    print(summarizier(tokenizer_ds["test"][i]["Text"],max_length=512))
    print(tokenizer_ds["test"][i]["Abstract"])

# #7.保存模型
# save_dir = "../T5"
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)