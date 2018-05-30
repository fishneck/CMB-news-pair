# -*- coding: utf-8 -*-
# Roc-J
'''
in bm25.py set parameters as follows
PARAM_K1 = 1.2
PARAM_B = 0.75
EPSILON = 0.25
'''
import jieba
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim import corpora
from gensim.summarization import bm25

def train_text():
    # 测试文档预处理
    test_doc = []
    test_datas = pd.read_csv("test_data.csv", encoding="gbk")
    test_titles = test_datas["title"]
    for title in test_titles:
        test_doc.append(title)
    test_doc_list = []
    for doc in test_doc:
        doc_list = [word for word in jieba.cut(doc)]
        test_doc_list.append(doc_list)

    # 训练集预处理，去除噪声
    train_doc = []
    datas = pd.read_csv("train_data.csv")
    train_titles = datas["title"]
    for title in train_titles:
        if 13<len(title)<500:
            doc_list = [word for word in jieba.cut(title)]
            all_doc_list.append(doc_list)
        else:
            all_doc_list.append("。")
    all_doc_list = []
    for doc in train_doc:
        doc_list = [word for word in jieba.cut(doc)]
        all_doc_list.append(doc_list)    
    
    
    # 制作词典
    dictionary = corpora.Dictionary(all_doc_list)
    dictionary.keys()
    print(dictionary.num_pos)
    dictionary.filter_extremes(no_below=25,no_above=0.5,keep_n=12330000)

    bm25Model = bm25.BM25(all_doc_list)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    print(average_idf)
    results = []
    for doc_test_list in test_doc_list:
        score = bm25.BM25.get_scores(bm25Model,doc_test_list,average_idf)
        similiar_sorted = sorted(enumerate(score), key=lambda item: -item[1])[:21]
        indexs = [str(item[0]+1) for item in similiar_sorted]
        results.append(" ".join(indexs))
    #写入文件
    with open("answers.txt", "w") as f:
        for item in results:
            item = item.strip().split()
            f.write("source_id" + "\t" + "target_id" + "\n")
            for i in range(1, 21):
                f.write(item[0] + "\t" + item[i] + "\n")

if __name__ == "__main__":
    train_text()
    print("welldone!")
