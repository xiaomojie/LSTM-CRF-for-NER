from model.data_tools import Dataset
from model.ner_model import NERModel
from model.setting import Config

if __name__ == "__main__":
    config = Config()
    config.mylog.write('\n---------- test.py 模型测试----------\n')
    print('---------- test.py 模型测试----------')

    # 模型建立
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # 创建测试集
    test = Dataset(config.filename_test, config.word_processing,
                   config.tag_processing)

    # 模型测试
    model.evaluate(test)
