import os
import tensorflow as tf


class BaseModel(object):
    def __init__(self, config):
        """
        初始化self.config 和 self.mylog
        :param config: config实例
        """
        self.config = config
        self.mylog = config.mylog
        self.sess = None
        self.saver = None

    def add_train_operation(self, lr, loss):
        """
         模型训练，梯度下降最小化损失loss
        :param lr: 学习率
        :param loss: 损失韩函数
        :return:
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(lr)  # Adam优化算法求解
            self.train_op = optimizer.minimize(loss)  # 最小化损失函数

    def initialize_session(self):
        """初始化会话"""
        self.mylog.write("初始化会话...\n")
        print("初始化会话...")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """
        加载会话和模型参数
        :param dir_model:  路径
        :return:
        """
        self.mylog.write("模型加载...\n")
        print("模型加载...")
        self.saver.restore(self.sess, dir_model)

    def close_session(self):
        """关闭会话"""
        self.sess.close()

    def save_model(self):
        """保存权重"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                                                 self.sess.graph)

    def train(self, train, dev):
        """
        模型训练
        :param train: 训练集（sentences, tags）
        :param dev: 验证集
        :return:
        """
        best_score = 0
        max_no_improve = 0  # 是否提前终止训练
        self.add_summary()  # tensorboard

        for epoch in range(self.config.nepochs):  # self.config.nepochs迭代次数
            self.mylog.write("第 %d 轮 / 共 %d 轮\n" % (epoch + 1,
                                                          self.config.nepochs))
            print("第 %d 轮 / 共 %d 轮" % (epoch + 1,
                                             self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)  # 返回的为f1值
            self.config.lr *= self.config.lr_decay  # 衰退学习率

            # 如果f1值超过三轮迭代没有更新，则停止训练
            if score >= best_score:
                max_no_improve = 0
                self.save_model()
                best_score = score

            else:
                max_no_improve += 1
                if max_no_improve >= self.config.max_no_improve:
                    self.mylog.write("-- f1值在 %d 轮中没变化，停止迭代\n" % max_no_improve)
                    print("-- f1值在 %d 轮中没变化，停止迭代" % max_no_improve)
                    break

    def evaluate(self, test):
        """
        使用测试集测试模型
        :param test: 测试集
        :return:
        """
        metrics = self.run_evaluate(test)
        msg = "\n".join(["{} {:04.2f}".format(k, v)
                         for k, v in metrics.items()])
        self.mylog.write(msg)
        print(msg)
