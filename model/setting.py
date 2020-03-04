from model.data_tools import *


class Config:
    def __init__(self, load=True):
        """
        初始化超参数和加载字典
        :param load: 若为真，则加载embedding
        """
        self.mylog = open(self.path_log, 'a', encoding='utf-8')  # 日志文件
        # load函数在数据集都生成之后设置为真，用来加载字典和embedding
        if load:
            self.load()

    def load(self):
        """
        加载字典,processing functions和预训练的embeddings
        在数据创建之后，这个函数会被执行
        :return:
        """
        # 字典
        self.words_vocab = load_vocab(self.filename_words)  # 获取到的是一个dict{ word --> id },从文件中加载字典，返回的是一个字典 dict{ word --> id }
        self.tags_vocab = load_vocab(self.filename_tags) # 从文件中加载tag字典，返回的是一个字典 dict{ word --> id }
        self.chars_vocabs = load_vocab(self.filename_chars)

        self.nwords = len(self.words_vocab)
        self.nchars = len(self.chars_vocabs)
        self.ntags = len(self.tags_vocab)

        # processing 函数: str -> id
        # 将单词（字符串）转换成一个列表或者元组
        # f("cat") = ([12, 4, 32], 12345) = (list of char ids, word id)
        self.word_processing = get_processing(self.words_vocab,
                                              self.chars_vocabs, lowercase=True, chars=True)
        self.tag_processing = get_processing(self.tags_vocab,
                                             lowercase=False, allow_unk=False)

        # 获取Embedding
        self.embeddings = (get_embedding(self.reduce_filename_embedding))

    dir_output = "../results/"  # 输出路径
    dir_model = dir_output + "model/"  # 模型保存路径
    path_log = dir_output + "log.txt"  # log日志

    # embeddings大小
    dim_word = 300
    dim_char = 100

    # embedding 文件名
    origin_filename_embedding = "../data/glove/glove_embedding_300dim.txt"
    # reduced embeddings
    reduce_filename_embedding = "../data/reduced_glove_300dimension.npz"

    # 数据集路径
    filename_dev = "../data/input/dev.txt"
    filename_test = "../data/input/test.txt"
    filename_train = "../data/input/train.txt"

    # 生成字典的路径
    filename_words = "../data/words.txt"
    filename_tags = "../data/tags.txt"
    filename_chars = "../data/chars.txt"

    # 参数
    nepochs = 13  # 迭代次数
    dropout = 0.5
    batch_size = 50  # batch大小

    lr = 0.001  # 学习率
    lr_decay = 0.9  # 衰退学习率

    max_no_improve = 3  # 若三次迭代没有对f1值进行更新，则提前终止

    # 模型超参数
    hidden_size_char = 100  # lstm for chars
    hidden_size_lstm = 300  # lstm for word embeddings
