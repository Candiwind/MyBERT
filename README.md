# MyBERT

imdb 数据集 下载地址:
https://ai.stanford.edu/~amaas/data/sentiment/

google bert-base-uncased 预训练模型下载地址：
https://huggingface.co/google-bert/bert-base-uncased/tree/main
有博客说只需要下载 config.json、pytorch_model.bin 和 vocab.txt
如果跑不通的话再下其他文件也行

下载完这两项之后，工程路径安排为：

MyBERT  |--python

        |--data     |--aclImdb_v1           |--(下载的数据)
        
        |--models   |--bert-base-uncased    |--(下载的预训练模型)

装完库后应该就能运行 main.py 了

