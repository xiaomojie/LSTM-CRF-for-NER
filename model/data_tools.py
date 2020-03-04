import numpy as np

# 全局变量
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = "ERROR: 该文件不存在： {}.".format(filename)
        super(MyIOError, self).__init__(message)


class Dataset(object):
    """数据处理"""

    def __init__(self, filename, processing_word=None, processing_tag=None):
        """
        :param filename: 文件路径
        :param processing_word: 以word作为输入的函数
        :param processing_tag: 以tag作为输入的函数
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.length = None

    def __iter__(self):
        """迭代产生一个tuple返回(words, tags)"""
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(words) != 0:
                        niter += 1
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """获取长度"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def load_vocab(filename):
    """
    从文件中加载字典，返回的是一个字典 dict{ word --> id }
    :param filename: 文件名
    :return: dict[word] = index 字典
    """
    try:
        dic = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):  # enumerate 返回的是 (index, word)
                word = word.strip()
                dic[word] = idx

    except IOError:
        raise MyIOError(filename)
    return dic


def get_embedding(filename):
    """
    获取词向量Embedding
    :param filename: 文件名
    :return: 词向量（matrix）
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]
    except IOError:
        raise MyIOError(filename)


def get_processing(words_vocab=None, chars_vocabs=None,
                   lowercase=False, chars=False, allow_unk=True):
    """
    将单词（字符串）转换成一个列表或者元组 （list, id）
    :param words_vocab: dict[word] = idx
    :param chars_vocabs: dict[word] = idx
    :param lowercase: 是否小写
    :param chars: 是否是字符
    :param allow_unk:
    :return:
    f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """

    def f(word):
        # 从词中获取字符
        if chars_vocabs is not None and chars:
            char_ids = []
            for char in word:
                # 忽略在字典之外的字符（chars.txt之外）
                if char in chars_vocabs:
                    char_ids += [chars_vocabs[char]]

        # 处理词
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 获取词的id
        if words_vocab is not None:
            if word in words_vocab:
                word = words_vocab[word]
            else:
                if allow_unk:
                    word = words_vocab[UNK]
                else:
                    raise Exception("Error：未知词，请检查字典！")

        # 返回元组 char ids, word id
        if chars_vocabs is not None and chars:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    填充子函数
    :param sequences:
    :param pad_tok: 填充字符
    :param max_length: 最大长度
    :return: 有相同长度的列表
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    填充函数
    :param sequences:
    :param pad_tok: 填充字符
    :param nlevels: pad 深度
    :return: 有相同长度的列表
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # 所有词的长度相同
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def get_minibatches(data, minibatch_size):
    """
    batches
    :param data: 数据集
    :param minibatch_size: (int)batch大小
    :return: 元组列表
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:  # minibatch_size = 20， 每20个提交一次
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    :param tok: token 的id
    :param idx_to_tag: dictionary {1: "I-per", ...}
    :return:
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            # 添加一个块
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # 终止条件
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
