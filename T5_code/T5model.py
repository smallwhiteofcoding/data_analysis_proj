# --------------------------导包--------------------------
import csv  #csv读取
import re   #正则表达
import torch #深度学习库
import random
import numpy as np
import pandas as pd
from rouge import Rouge #用于评估摘要质量的库
from tqdm import tqdm #用于生成进度条
from transformers import pipeline #用于生成摘要的库
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #用于生成摘要的库

# 方便复现
def set_seed(seed):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置CUDA的随机种子 ，CUDA是一种用于加速深度学习计算的硬件加速器，通常与NVIDIA的GPU配合使用
    torch.cuda.manual_seed(seed)
    # 设置CUDA的随机数生成器为确定性
    torch.backends.cudnn.deterministic = True
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python的随机种子
    random.seed(seed)

# 评测 rouge_L 分数:评估自动生成的摘要与参考摘要之间的相似度
def print_rouge_L(output, label): # 输入 结果output 和 参考文本label
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, label)# 使用 Rouge 库计算 rouge_L 分数

    rouge_L_f1 = 0    # 精确率和召回率的调和平均数
    rouge_L_p = 0     # 召回率：LCS（最长公共子序列）长度与参考摘要长度的比值，检验完整性
    rouge_L_r = 0      #精确率：LCS长度与模型生成摘要长度的比值，检验精确度
    # 遍历 rouge_score
    for d in rouge_score:
        # 计算 rouge_L 的 fpr
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    # 打印 rouge
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))

set_seed(0)



# --------------------------读取数据--------------------------


train_path = '../T5_datasets/train_dataset.csv'  # 自定义训练集路径
train= pd.read_csv(train_path, sep='\t', names=["Index", "Text", "Abstract"])

test_path = '../T5_datasets/test_dataset.csv'  # 自定义测试集路径
test = pd.read_csv(test_path, sep='\t', names=["Index", "Text"])



# --------------------------加载【T5】模型--------------------------
#基于Transformer的模型，主要用于自然语言生成（NLG）任务

#处理文本中的空白字符，将多个空白字符替换为一个空格，并将文本中的换行符替换为一个空格
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

model_name = "T5"

tokenizer = AutoTokenizer.from_pretrained(model_name) #构建分词器
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) #加载T5模型



# # --------------------------查看单条样本预测结果-------------------------
# # 索引第i条样本
# i = 0
# # 定义article_text，用于存储第i条样本的正文
# article_text = train["Text"][i]
# # 定义article_abstract，用于存储第i条样本的摘要
# article_abstract = train["Abstract"][i]
#
# # 使用tokenizer对第i条样本的正文进行编码，返回结果为input_ids
# input_ids = tokenizer(
#     [WHITESPACE_HANDLER(article_text)],
#     return_tensors="pt",  #pytorch格式数据
#     padding="max_length",  #填充到最大长度
#     truncation=True,
#     max_length=512
# )["input_ids"]
#
# # 使用model对input_ids进行生成，生成结果为output_ids
# output_ids = model.generate(
#     input_ids=input_ids,    #输入文本
#     max_length=512,
#     min_length=int(len(article_text)/32), #生成文本的最小长度，默认为输入文本长度的1/32。
#     no_repeat_ngram_size=3,   #确保重复单词小于3个
#     num_beams=5              #设置候选文本数量
# )[0]
#
# # 使用tokenizer对output_ids进行解码，得到生成的摘要
# summary = tokenizer.decode(
#     output_ids,
#     skip_special_tokens=True,   #清除特殊token
#     clean_up_tokenization_spaces=False  #保留空格
# )
#
# # 打印生成的摘要
# print(f"Generate：\n{summary}")
# # 打印标签（即真实摘要）
# print(f"Label：\n{article_abstract}")
# # 使用print_rouge_L函数计算生成的摘要与标签之间的Rouge-L分数
# print_rouge_L(summary,article_abstract)



# # --------------------------训练-------------------------
#
# multi_sample =500  #训练集数量
# sumaries =list()  #摘要结果
# # 使用tqdm显示进度条
# for idx,article_text in tqdm(enumerate(train["Text"][0:500]),total=multi_sample):
#     # 使用tokenizer对输入文本进行编码
#     input_ids = tokenizer(
#     [WHITESPACE_HANDLER(article_text)],
#     return_tensors="pt",
#     padding="max_length",
#     truncation=True,
#     max_length=512
#     )["input_ids"]
#
#     # 使用模型生成输出文本
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=512,
#         min_length=int(len(article_text)/32),
#         no_repeat_ngram_size=3,
#         num_beams=5
#     )[0]
#
#     # 对输出文本进行解码
#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     # 将生成的摘要添加到训练数据集中
#     train.loc[idx,"summary"] = summary
#     print(idx+500,summary)
#     # 将生成的摘要添加到sumaries列表中
#     sumaries.append([idx+500,summary])
#
# # 创建csv文件路径
# csv_file_path = 'datasets/submit.csv'
# # 打开csv文件
# with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
#     # 创建csv写入对象
#     write = csv.writer(csvfile)
#     # 将sumaries列表写入csv文件
#     write.writerows(sumaries)

# --------------------------预测-------------------------
# 定义一个空列表，用于存储预测结果
sumaries =list()
# 使用tqdm函数，对测试集进行迭代，enumerate函数返回索引和值
for idx, article_text in tqdm(enumerate(test["Text"]), total=1000):
    # 使用tokenizer函数，对文章进行编码
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=768
    )["input_ids"]

    # 使用model.generate函数，生成摘要
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=512,
        min_length=int(len(article_text) / 32),
        no_repeat_ngram_size=3,
        num_beams=5
    )[0]

    # 使用tokenizer.decode函数，将摘要解码
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # 将预测结果添加到列表中
    sumaries.append([idx,summary])

# 定义保存结果的文件路径
csv_file_path = '../T5_datasets/submit1.csv'
# 使用with open函数，打开文件，并指定编码格式
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    # 使用write函数，将预测结果写入文件
    write = csv.writer(csvfile)
    # 使用writerows函数，将列表写入文件
    write.writerows(sumaries)

#重写文件：改写为提交文件格式
import csv
with open('../T5_datasets/submit1.csv', 'r', newline='', encoding='utf-8') as f1:
    with open('T5_datasets/newsub.csv', 'w', newline='', encoding='utf-8') as f2:
        reader = csv.reader(f1)
        w = csv.writer(f2,delimiter='\t')
        for row in reader:
            row[1]=str(row[1])
            print(row[1])
            w.writerow(row)



