import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sqlite3
import time

constructed_corpus_path = "dialogue_corpus.db"
trained_svr_path = "user_acceptance_estimation.pkl"

def get_all_usertalk_data_from_corpus():
    """
    コーパスから、ユーザ発話のデータを全て取得する
    talker text, talk_content text, emotion text, acceptance text
    """
    corpus = sqlite3.connect(constructed_corpus_path)
    cur = corpus.cursor()
    scenes = ["cleaning", "exercise", "game", "lunch", "sleep"]
    dialogue_idx = [str(i) for i in range(200)]
    all_usertalk_data = [] # コーパス内の全てのユーザ発話のデータ
    all_talk_data = [] # コーパス内の全ての発話（ユーザとシステム両方）のデータ
    append_ok_idx_list = []
    append_flag = True
    
    t1 = time.time()
    count = 0
    for scene in scenes:
        for idx in dialogue_idx:
            dialogue_name = scene + idx
            # 一般的なインジェクション攻撃はテーブル名を変える攻撃が無いから、?で置き換えるやつが実装されていない（かも）
            # テーブル名を変える攻撃の場合は文字列の長さで防げるので、formatで大丈夫
            # "select * from {}".format(table)にすると、tableをA where '1'='1'にすれば攻撃できるけど
            # 存在するテーブル名の長さよりも長ければ攻撃だと判定できる（len(table) > (存在するテーブル名): 攻撃だと判定）
            # by.師匠
            cur.execute("select * from {}".format(dialogue_name))
            a_dialogue = cur.fetchall() # fetchall() -> 1対話に含まれるユーザ発話のデータを取得している
            for i, talk_data in enumerate(a_dialogue):
                if talk_data[0] == "user" : # ユーザの発話データを見ている場合
                    if talk_data[3] != "NONE":
                        all_usertalk_data.append(talk_data) # ユーザ発話のデータをappend
                        append_ok_idx_list.append(count)
                count += 1
            """
            for i, talk_data in enumerate(a_dialogue):
                count += 1
                if talk_data[0] == "user": # ユーザの発話データを見ている場合
                    if talk_data[3] != "NONE":
                        all_usertalk_data.append(talk_data) # ユーザ発話のデータをappend
                    else:
                        all_talk_data.pop()
                        append_flag = False
                if append_flag:
                    if i + 1 < len(a_dialogue):
                        if a_dialogue[i + 1][0] == "system" and talk_data[0] == "system":
                            pass
                        else:
                            append_ok_idx_list.append(count)
                            all_talk_data.append(talk_data)
                    elif talk_data[0] == "user":
                        append_ok_idx_list.append(count)
                        all_talk_data.append(talk_data)
                else:
                    append_flag = True
                    if i + 2 < len(a_dialogue):
                        if a_dialogue[i + 2][0] == "system":
                            append_flag = False
            """
    
    t2 = time.time()
    print("Finished getting all user talks from {}: about {} seconds.".format(constructed_corpus_path, t2 - t1))
    print("in def_usertalks len(append_ok_idx_list) = {}".format(len(append_ok_idx_list)))
    return all_usertalk_data[:], append_ok_idx_list # 下記SVRの訓練のy, Xの特徴ベクトルのキーに該当する

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
    tmp_X = [] # "NONE"が含まれているbag of wordsの特徴ベクトル（2次元のリスト）
    all_usertalk_data, append_ok_idx_list = get_all_usertalk_data_from_corpus()
    with open("./svr_bow_vec.txt") as file:
        tmp_X = eval(file.read())[:]
    t2 = time.time()
    print("Finished loading input bag of words vector: about {} seconds.".format(t2 - t1))
    
    t1_2 = time.time()
    X = []
    print("len(tmp_X) = {}".format(len(tmp_X)))
    for i in append_ok_idx_list:
        X.append(tmp_X[i])
    t2_2 = time.time()
    print("len(X[0]) = {}".format(len(X[0])))
    print("len(X) = {}".format(len(X)))
    print("Finished making input bag of words vector except NONE label: about {} seconds.".format(t2_2 - t1_2))
    
    t3 = time.time()
    y = [] # SVRの出力：ユーザの受諾度合いの正解ラベル [acceptance]
    print("in def_svr ... len(all_usertalk_data) = {}".format(len(all_usertalk_data)))
    for usertalk_data in all_usertalk_data:
        an_acceptance_label = usertalk_data[3]
        print("an_acceptance_label = {}".format(an_acceptance_label))
        y.append(int(an_acceptance_label))
    t4 = time.time()
    print("Finished making output vector: about {} seconds.".format(t4 - t3))
    
    print("yのサイズ：{}".format(len(y)))
    
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