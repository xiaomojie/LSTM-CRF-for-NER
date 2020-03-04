from model.data_tools import *
from model.setting import Config

config = Config(load=False)  # load 为假，此时字典还没创建好，不能加载
mylog = config.mylog


def read_dataset(filename):
    """
    从数据集中读取数据
    :param filename: 文件名
    :return: 以列表的形式返回数据
    """
    with open(filename) as f:
        words, tags = [], []
        temp_words, temp_tags = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0:  # 一个句子已结束
                if temp_words is not []:
                    words.append(temp_words)
                    tags.append(temp_tags)
                    temp_words, temp_tags = [], []
            else:
                items = line.split(' ')
                word, tag = items[0], items[-1]
                temp_words += [word]
                temp_tags += [tag]
    return words, tags


def create_vocabs(datasets):
    """
    从数据集中创建字典
    :param datasets: 数据集列表  [[words,tags],[words,tags],[words,tags]]
    :return: 在数据集中出现过的word列表和tags列表
    """
    mylog.write("从数据集中创建 word 字典...\n")
    print("从数据集中创建 word 字典...")
    words_vocab = set()
    tags_vocab = set()
    for dataset in datasets:  # [words,tags]
        words = dataset[0]
        tags = dataset[1]
        for word in words:
            words_vocab.update(word)  # update函数list添加到set中
        for tag in tags:
            tags_vocab.update(tag)
    return words_vocab, tags_vocab


def create_embedding_vocab(filename):
    """
    从预训练的embedding中加载vocab
    :param filename: Embedding文件名
    :return: 字典set()
    """
    mylog.write("从预训练的词嵌入中创建字典...\n")
    print("从预训练的词嵌入中创建字典...")
    vocab = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    return vocab


def write_vocab(vocab, filename):
    """
    将词典写入文件，一个词一行
    :param vocab: 词典
    :param filename: 文件名
    :return:
    """
    mylog.write("写文件: %s ...\n" % filename)
    print("写文件: %s ..." % filename)
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("-- 已完成. 写入数量: %d" % (len(vocab)))
    mylog.write("-- 已完成. 写入数量: %d\n" % (len(vocab)))


def write_embedding(vocab, embedding_filename, reduced_embedding_filename, dim):
    """
    从预训练的词嵌入中读取embedding，存入到数组中，然后保存
    :param vocab: 字典 vocab[word] = index （从words.txt中读取的）
    :param embedding_filename: 词嵌入文件
    :param reduced_embedding_filename: 保存有用的词嵌入的文件
    :param dim: (int) embeddings的维度
    :return:
    """
    mylog.write("获取并保存词嵌入：%s...\n" % reduced_embedding_filename)
    print("获取并保存词嵌入：%s..." % reduced_embedding_filename)
    embeddings = np.zeros([len(vocab), dim])
    with open(embedding_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            # 训练、验证、测试数据集中有的词才保存到数组中，这样可以减少look_up_embedding 所需时间
            if word in vocab:
                embedding = [float(x) for x in line[1:]]
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)  # word -> index -> embedding

    np.savez_compressed(reduced_embedding_filename, embeddings=embeddings)


def create_char_vocab(dataset):
    """
    利用训练集创建 character 词典
    :param dataset: 数据集
    :return: 数据集中所有的字符
    """
    mylog.write("从数据集中创建 character 字典...\n")
    print("从数据集中创建 character 字典...")
    chars_vocab = set()
    words = dataset[0]
    for word in words:
        # 由于word是一个字符串而不是一个列表，所以会将word分成一个一个的char进行加入，并去重
        for chars in word:
            chars_vocab.update(chars)

    return chars_vocab


if __name__ == "__main__":
    """
    创建数据,使用整个数据集（训练集，验证集， 测试集）来获取字典（words, tags, characters）
    从预训练的词嵌入中读出有用的word embedding保存，创建char字典
    :return:
    """
    mylog.write('---------- build_vocab.py 创建字典----------\n')

    # 获取数据集
    dev = read_dataset(config.filename_dev)  # [words,tags]
    test = read_dataset(config.filename_test)
    train = read_dataset(config.filename_train)

    # 从数据集中创建 Word 词典和 Tag 词典
    words_vocab, tags_vocab = create_vocabs([train, dev, test])
    # 从预训练的词嵌入文件中获取其中包含的所有的词
    glove_vocab = create_embedding_vocab(config.origin_filename_embedding)

    vocab = words_vocab & glove_vocab  # 取交集
    vocab.add(UNK)
    vocab.add(NUM)

    # 保存Word 词典和 Tag 词典
    write_vocab(vocab, config.filename_words)
    write_vocab(tags_vocab, config.filename_tags)

    # 从word.txt中加载字典
    vocab = load_vocab(config.filename_words)  # 获取到的是一个dict{ word --> id }
    # 从Embedding 中获取有用的保存起来
    write_embedding(vocab, config.origin_filename_embedding, config.reduce_filename_embedding, config.dim_word)

    # 训练集
    train = read_dataset(config.filename_train)
    # 利用训练集创建 character 词典
    chars_vocabs = create_char_vocab(train)
    # 保存
    write_vocab(chars_vocabs, config.filename_chars)
