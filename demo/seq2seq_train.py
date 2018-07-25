from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras_text_summarization.library.applications.fake_news_loader import \
    fit_text
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.utility.plot_utils import \
    plot_and_save_history

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42) # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    data_dir_path = './data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/論文前201.csv")

    print('extract configuration from input texts ...')
    Y = df.title
    X = df['text']

    config = fit_text(X, Y) # call line7 from keras_text_summarization.library.applications.fake_news_loader import fit_text

    summarizer = Seq2SeqSummarizer(config) # 將config回傳參數放入seq2seq.py程式中

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=10)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()
