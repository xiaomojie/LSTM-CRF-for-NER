from model.data_tools import Dataset
from model.ner_model import NERModel
from model.setting import Config

if __name__ == "__main__":
    config = Config()  # 此时参数load为默认值True

    config.mylog.write('\n---------- train.py 模型训练----------\n')
    print('---------- train.py 模型训练----------')

    # 模型建立
    model = NERModel(config)
    model.build()

    # 训练集与验证集
    dev = Dataset(config.filename_dev, config.word_processing, config.tag_processing)
    train = Dataset(config.filename_train, config.word_processing, config.tag_processing)

    # 模型训练
    model.train(train, dev)
