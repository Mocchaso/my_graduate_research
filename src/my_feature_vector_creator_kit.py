## 全てmy_feature_vector_creator.ipynbで使用。

from my_data_preparation_kit import convert_dict_to_json, load_json_as_dict, save_dict_as_json, normalize, get_all_usertalk_data_from_corpus
import sqlite3
import MeCab
import time
import scipy.sparse as sci_sp

constructed_corpus_path = "dialogue_corpus.db"
pn_ja_dict_path = "../pn_ja.dic"
svr_bow_path = "svr_bow_vec.npz"

def get_all_words_in_corpus():
    """
    コーパス内の発話（ユーザ発話とシステム発話両方）に含まれる単語を全て取得する
    """
    corpus = sqlite3.connect(constructed_corpus_path)
    cur = corpus.cursor()
    scenes = ["cleaning", "exercise", "game", "lunch", "sleep"]
    dialogue_idx = [str(i) for i in range(200)]
    m = MeCab.Tagger("-Owakati")
    all_words_list = []
    
    t1 = time.time()
    for scene in scenes:
        for idx in dialogue_idx:
            dialogue_name = scene + idx
            # 一般的なインジェクション攻撃はテーブル名を変える攻撃が無いから、?で置き換えるやつが実装されていない（かも）
            # テーブル名を変える攻撃の場合は文字列の長さで防げるので、formatで大丈夫
            # "select * from {}".format(table)にすると、tableをA where '1'='1'にすれば攻撃できるけど
            # 存在するテーブル名の長さよりも長ければ攻撃だと判定できる（len(table) > (存在するテーブル名): 攻撃だと判定）
            # by.師匠
            cur.execute("select * from {}".format(dialogue_name))
            a_dialogue = cur.fetchall() # fetchall() -> 1対話に含まれる全ての発話データを取得している
            for i, talk_data in enumerate(a_dialogue):
                talk_content = normalize(talk_data[1])
                words_in_talk_content = m.parse(talk_content).split()
                all_words_list.extend(words_in_talk_content) # コーパス内の全発話内の単語を全て記録するリストに連結する
    t2 = time.time()
    print("Finished getting all words from {}: about {} seconds.".format(constructed_corpus_path, t2 - t1))
    return all_words_list[:]

def get_key_from_values(dic, val):
    """
    引数valで指定した辞書の値からキーを抽出する（1個のみ）
    参考サイト：https://note.nkmk.me/python-dict-get-key-from-value/
    """
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return None

def make_svr_learning_feature(w_table, s_table):
    """
    ユーザが入力した発話から特徴ベクトルを作成する
    
    手順
    1. MeCabを用いて形態素解析 -> ユーザ発話から言語特徴量を抽出する
       言語特徴量としては ... 当該発話に含まれる単語と、WordNetを用いて抽出した同義語からなるbag of wordsの単語特徴ベクトルを用意する
       SVRの学習特徴量としては ... コーパスに含まれる全単語数の次元数のベクトル
                                   入力された発話に含まれる単語に相当する次元を1、含まれない単語に相当する次元を0とする疎ベクトルを作成する
       参考サイト：https://www.pytry3g.com/entry/2018/03/21/181514
    2. 単語極性辞書 -> 発話文に含まれる単語に付与された極性スコアの平均値を算出する
       この2の値を上記の単語特徴ベクトルに付加 -> 1発話の特徴ベクトルとする
    
    bag of wordsの最終的な構造：(row, col) = (ユーザ発話, コーパスに含まれる全ての単語 + 各発話の極性スコアの平均値)
    """
    # 単語極性辞書の読み込み
    pn_table = {}
    with open(pn_ja_dict_path) as file:
        for line in file:
            line = line.split(":")
            pn_table[line[0]] = float(line[3])
    print("Finished loading pn_ja.dic.")
    
    # コーパス内の全てのユーザ発話のデータ：[ talk_data -> tuple, ... ]
    all_usertalk_data = get_all_usertalk_data_from_corpus()
    # コーパス内の全単語
    all_words_in_corpus = get_all_words_in_corpus()
    # コーパス内の全単語の重複無しバージョン（順序を保持している）
    # 参考サイト：https://note.nkmk.me/python-list-unique-duplicate/
    all_words_in_corpus_uniq = sorted(set(all_words_in_corpus), key = all_words_in_corpus.index)
    print("Finished making all_words-list which removed duplication. (order of the list is kept)")
    
    # コーパス内の全単語とその同義語を、対応する列にまとめたリスト
    # 対応する単語IDが無い or 同義語が無い場合は、その単語自体のみを含んだリストが格納される
    t0 = time.time()
    columns = []
    for word in all_words_in_corpus_uniq:
        column = [word]
        wordids = []
        if word in w_table.keys(): # wordがユーザ発話に含まれる単語ならば
            wordids = w_table[word]
        if wordids != []: # wordに対応する単語IDがあれば
            for wordid in wordids:
                # wordidに対応する同義語をcolumnに追加
                synonyms = s_table[str(wordid)]
                column.extend(synonyms)
        columns.append(column)
    t01 = time.time()
    print("Finished making a list about words in corpus and their synonyms: about {} seconds.".format(t01 - t0))
    
    # bag of wordsを作る
    t1 = time.time()
    bow_set = []
    m = MeCab.Tagger("-Owakati")
    for usertalk_data in all_usertalk_data:
        if usertalk_data[3] != "NONE": # NONEとアノテートされている受諾度合いラベルは、学習データから除く
            bow = [0] * len(columns) # 1発話の特徴ベクトル
            word_pn_scores = 0 # 1発話当たりの極性スコアの記録
            usertalk_content = normalize(usertalk_data[1]) # ユーザ発話の内容
            words_in_usertalk_content = m.parse(usertalk_content).split() # ユーザ発話に含まれる単語のリスト
            for word in words_in_usertalk_content:
                # word：ユーザ発話に含まれる単語
                # columns：コーパス内の全単語とその同義語を、対応する列にまとめたリスト
                for col_i, column in enumerate(columns):
                    if word in column: # wordが、対応する列にまとめておいた単語と同義語のリストに含まれていたら
                        bow[col_i] = 1

                if word in pn_table.keys(): # 1つのユーザ発話内の単語の極性スコアを加算
                    word_pn_scores += pn_table[word]
            else:
                # 1発話の単語について全て調べ終わったら、発話文の各単語の極性スコアの平均値を算出する
                # 小数点以下切り捨て防止のため分子にfloat()、0除算防止のため分母に+1
                word_pn_ave = float(word_pn_scores) / (len(words_in_usertalk_content) + 1)
                bow.append(word_pn_ave)
            bow_set.append(bow)
    t2 = time.time()
    print("Finished making bag of words vector: about {} seconds.".format(t2 - t1))
    
    # 作成したbag of wordsを疎行列に変換する
    # 参考サイト：
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz
    # http://hamukazu.com/2014/09/26/scipy-sparse-basics/
    # https://ohke.hateblo.jp/entry/2018/01/07/230000
    t3 = time.time()
    bow_set = sci_sp.lil_matrix(bow_set)
    bow_set = bow_set.tocsr() # csr_matrixに変換する -> ファイルに保存できる＆行単位に高速にアクセスできる
    t4 = time.time()
    print("Finished converting bag of words vector to sparse matrix: about {} seconds.".format(t4 - t3))
    print("shape of bag of words vector: {}".format(bow_set.shape))
    # bag of wordsをnpzファイルに保存する
    t5 = time.time()
    sci_sp.save_npz(svr_bow_path, bow_set)
    t6 = time.time()
    print("Finished saving bag of words vector at {}: about {} seconds.".format(svr_bow_path, t6 - t5))