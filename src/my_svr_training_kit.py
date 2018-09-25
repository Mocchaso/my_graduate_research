import scipy.sparse as sp
from sklearn.svm import SVR
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sqlite3
import time
from my_data_preparation_kit import get_all_usertalk_data_from_corpus
from my_feature_vector_creator_kit import get_all_words_in_corpus
import struct

constructed_corpus_path = "dialogue_corpus.db"
svr_bow_bin_path = "svr_bow_vec.bin" # バイナリデータ化したbag of wordsの保存先のパス（ファイル名）
trained_svr_path = "user_acceptance_estimation.pkl"

def train_svr():
    """
    ユーザの受諾度合いの推定のために、SVRのモデルを学習させる
    学習データ ... 入力：ユーザ発話の特徴ベクトル、出力：ユーザの受諾度合いの正解ラベル
    正解ラベル -> 発話に付与される5段階のユーザの受諾度合いラベル
    
    参考サイト：
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
    https://web-salad.hateblo.jp/entry/2014/11/09/090000
    """
    t1 = time.time()
    all_usertalk_data = get_all_usertalk_data_from_corpus()
    print("len(all_usertalk_data) = {}".format(len(all_usertalk_data)))
    
    # バイナリデータとして書き込んだbag of wordsを読み込んで復元する
    X = None # bag of wordsの特徴ベクトル（2次元のリスト）
    with open(svr_bow_bin_path, "rb") as file:
        read_bow_set = file.read()
        bow_set_bytes_tuple = struct.unpack("s" * len(read_bow_set), read_bow_set) # 読み込んだバイナリデータをbytesとして復元する
        bow_set_unicode = list(map(lambda x: x.decode(), bow_set_bytes_tuple)) # bytesからunicode文字列に変換する
        # リスト内の文字を結合 -> 二次元リストに変換 -> 疎行列として扱う
        # (row, col) = (10983, 6548) -> (全ユーザ発話の合計, コーパスに含まれる全単語数 + 1)
        X = sp.lil_matrix(eval("".join(bow_set_unicode))[:])
    t2 = time.time()
    print("Finished loading input vector, bag of words: about {} seconds.".format(t2 - t1))
    
    t3 = time.time()
    y = [] # SVRの出力：ユーザの受諾度合いの正解ラベル [acceptance]
    for usertalk_data in all_usertalk_data:
        if usertalk_data[3] != "NONE": # NONEとアノテートされている受諾度合いラベルは、学習データから除く
            an_acceptance_label = int(usertalk_data[3])
            y.append(an_acceptance_label)
    t4 = time.time()
    print("Finished making output vector, user acceptance labels: about {} seconds.".format(t4 - t3))
    
    print("len(y) = {}".format(len(y)))
    
    # モデルの学習
    t5 = time.time()
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X, y)
    t6 = time.time()
    print("Finished training SVR model: about {} seconds.".format(t6 - t5))
    
    # 予測モデルをシリアライズ（訓練済みのモデルをファイルに保存）
    t7 = time.time()
    joblib.dump(clf, trained_svr_path)
    t8 = time.time()
    print("Finished saving SVR model as {}: about {] seconds.".format(trained_svr_path, t8 - t7))
    
    print("Finished training and saving SVR model!")