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

train_dataset = train_dataset.train_test_split(100, seed=42)#将训练数据划分为训练集和验证集

#处理文本中的空白字符，将多个空白字符替换为一个空格，并将文本中的换行符替换为一个空格
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# 1.模型加载

model_name = "../T5"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2.分词与张量化
def process_func(examples):
    # 'input_ids'是一个PyTorch张量，包含了分词后的标记ID
    inputs = tokenizer(examples['Text'], max_length=512, truncation=True)
    # 将分词器的输入设置为目标分词器的输出
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Abstract"], max_length=512, truncation=True)
    # 将输入的标签设置为与输入的文本长度相同的序列
    inputs["labels"] = labels["input_ids"]
    # 返回处理后的输入
    return inputs

tokenizer_ds = train_dataset.map(process_func, batched=True)
print(tokenizer_ds)


#3.创建模型
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

rouge = Rouge()

#4.创建评估函数
def compute_metric(evalPred):

    # 从 evalPred 中解码出预测结果和实际结果
    predictions,labels=evalPred
    # 使用 tokenizer 解码预测结果
    output = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将实际结果中的非 -100 值替换为 tokenizer.pad_token_id
    labels = np.where(labels != -100,labels,tokenizer.pad_token_id)
    # 使用 tokenizer 解码实际结果
    d_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 使用 Rouge 库计算 rouge_L 分数
    rouge_score = rouge.get_scores(output, d_labels,avg=True)
    # 返回 rouge-1、rouge-2 和 rouge-l 的分数
    return {
        'rouge-1': rouge_score["rouge-1"]['f'],
        'rouge-2': rouge_score["rouge-2"]['f'],
        'rouge-l': rouge_score["rouge-l"]['f'],
    }




#5.配置训练参数,微调
# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    # 输出目录
    output_dir='../T5',
    # 批处理大小
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    # 打印的步骤
    logging_steps=8,
    # 评估策略
    evaluation_strategy="epoch",
    # 保存策略
    save_strategy="epoch",
    # 最佳模型指标
    metric_for_best_model="rouge-l",
    # 是否使用生成进行预测
    predict_with_generate=True,
)


#6.创建训练模型
# 定义Seq2SeqTrainer对象，用于训练模型

trainer = Seq2SeqTrainer(
    # 传入训练参数
    args=training_args,
    # 传入模型
    model=model,
    # 传入分词器
    tokenizer=tokenizer,
    # 传入训练数据集
    train_dataset=tokenizer_ds["train"],
    # 传入测试数据集
    eval_dataset=tokenizer_ds["test"],
    # 传入评估指标
    compute_metrics=compute_metric,
    # 传入数据collator
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)
trainer.train()


#6.预测
summarizier = pipeline('summarization', model=model, tokenizer=tokenizer)
for i in range(5):
    print(summarizier(tokenizer_ds["test"][i]["Text"],max_length=512))
    print(tokenizer_ds["test"][i]["Abstract"])

