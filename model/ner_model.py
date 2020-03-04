import numpy as np
import tensorflow as tf

from model.data_tools import get_minibatches, pad_sequences, get_chunks
from .base_model import BaseModel


class NERModel(BaseModel):
    """定义用于命名实体识别的一些函数"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in  # 获取到各个tag的id
                           self.config.tags_vocab.items()}

    def add_placeholders(self):
        """占位符定义"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        填充数据，并建立一个feed字典
        :param words: 句子列表。 一个句子是一个单词列表的ID列表。 一个字是一个ID列表
        :param labels: ids 列表
        :param lr: 学习率
        :param dropout: 保持概率
        :return:  dict {placeholder: value}
        """
        # 填充数据
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                               nlevels=2)

        # 创建feed字典
        feed = {self.word_ids: word_ids, self.sequence_lengths: sequence_lengths, self.char_ids: char_ids,
                self.word_lengths: word_lengths}

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_embeddings_operation(self):
        """生成self.word_embeddings = [word_embedding, char_embedding]"""
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                self.config.embeddings,  # 初始值
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=False)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,  # 在_word_embeddings中查找word_ids
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            # 获取 char embeddings 矩阵
            _char_embeddings = tf.get_variable(  # 此时的_char_embeddings还没有生成
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[self.config.nchars, self.config.dim_char])  # 定义为[nchars, dim_char][char.txt文件中的数量，dim_char]
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     self.char_ids, name="char_embeddings")

            # 将time dimension 放到 axis=1
            s = tf.shape(char_embeddings)
            # shape = (batch x sentence, word, dim of char embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                         shape=[s[0] * s[1], s[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

            # 使用bi-lstm 处理字符chars， 生成字符embedding
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,  # 先向RNN
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,  # 后向RNN
                                              state_is_tuple=True)
            # 双向rnn， 返回一个tuple(outputs, outputs_states), 其中,outputs是一个tuple(outputs_fw, outputs_bw)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output 将双向输出合并
            _, ((_, output_fw), (_, output_bw)) = _output  # 最后一个time的输出状态
            # shape = (batch x sentence, 2 x char_hidden_size)
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            # shape = (batch, sentence, 2 x char_hidden_size)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * self.config.hidden_size_char])
            # 将word embedding 与char embedding合并  w = [wglove,wchars] ∈ Rn
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_lstm_operation(self):
        """
        定义self.logits，将联合向量w = [wglove,wchars]作为lstm的输入，
        全连接层计算每类的得分
        """
        # 将 pretrained_embedding 和 char_pre 联合起来的向量作为lstm的输入
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            """
            返回值：
                一个(outputs, output_states)的元组
                其中，
                1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。假设
                time_major=false,tensor的shape为[batch_size, max_time, depth]。实验中使用tf.concat(outputs, 2)将其拼接。
                2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
                output_state_fw和output_state_bw的类型为LSTMStateTuple。
                LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
            """
            output = tf.concat([output_fw, output_bw], axis=-1)  # 所有的中间状态的输出
            output = tf.nn.dropout(output, self.dropout)

        # 全连接计算一个分数，每一维代表输入属于该类的得分
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_crf_operation(self):
        """
        定义损失函数
        :return:
        """
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding 记录转移过程
        self.loss = tf.reduce_mean(-log_likelihood)

        tf.summary.scalar("loss", self.loss)

    def build(self):
        """创建模型"""

        self.add_placeholders()  # 添加占位符

        # 生成self.word_embeddings = [word_embedding, char_embedding]
        self.add_embeddings_operation()

        # 将 pretrained_embedding 和 char_pre 联合起来的向量作为lstm的输入
        # 全连接计算一个分数，每一维代表输入属于该类的得分
        self.add_lstm_operation()

        # 定义损失函数  crf
        self.add_crf_operation()

        # 添加训练操作并初始化会话
        self.add_train_operation(self.lr, self.loss)  # lr: 学习率， loss： 损失函数
        self.initialize_session()

    def predict_batch(self, words):
        """
        :param words: 句子列表
        :return:
            labels_pred: 预测出的类别标签
            sequence_length: 长度
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        # 获取tag得分，获取CRF的转移参数(transition params)
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
            [self.logits, self.trans_params], feed_dict=fd)

        # vitervi_decode模型
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """
        在训练集上执行一次完整的训练，并在验证集上评估
        :param train: 训练集 （sentences, tags）
        :param dev: 验证集
        :param epoch: 当前迭代轮数
        :return: f1: float 型
        """
        batch_size = self.config.batch_size  # 20
        nbatches = (len(train) + batch_size - 1) // batch_size

        # 在数据集上进行迭代
        for i, (words, labels) in enumerate(get_minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,  # 获取到一个feed dict
                                       self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = ", ".join(["{} {:04.2f}".format(k, v)
                         for k, v in metrics.items()])
        self.mylog.write(msg + '\n')
        print(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        """
        在数据集上进行评估
        :param test: 验证集(sentences, tags)
        :return: (dict) metrics{accuracy, precision, recall, f1}
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in get_minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.tags_vocab))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.tags_vocab))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0  # precision
        r = correct_preds / total_correct if correct_preds > 0 else 0  # recall
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"accuracy": 100 * acc, "precision": 100 * p, "recall": 100 * r, "f1": 100 * f1}

    def predict(self, words_raw):
        """
        模型预测
        :param words_raw: 句子中的词列表
        :return: 句子中词的tags
        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
