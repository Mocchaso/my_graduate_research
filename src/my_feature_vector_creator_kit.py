## 全てmy_feature_vector_creator.ipynbで使用。

from my_data_preparation_kit import convert_dict_to_json, load_json_as_dict, save_dict_as_json, get_all_usertalks_from_corpus, normalize
import sqlite3
import MeCab
import time

constructed_corpus_path = "dialogue_corpus.db"
pn_ja_dict_path = "../pn_ja.dic"

def get_analyzed_all_usertalks():
    """
    分かち書きしたユーザ発話を全て取得する
    """
    all_usertalk = get_all_usertalks_from_corpus()
    m = MeCab.Tagger("-Owakati")
    
    result = []
    t1 = time.time()
    for i, usertalks in enumerate(all_usertalk): # usertalks：1対話それぞれにおけるユーザ発話を格納しているリスト
        for usertalk in usertalks:
            usertalk = normalize(usertalk) # 発話の正規化
            wakati = m.parse(usertalk) # 分かち書きモードで形態素解析
            result.append(wakati)
    t2 = time.time()
    
    print("Finished getting all analyzed user talks: about {} seconds.".format(t2 - t1))
    # イメージ
    # ['私 は ラーメン が 好き です 。',
    #  '私 は 餃子 が 好き です 。',
    #  '私 は ラーメン が 嫌い です 。']
    return result

def get_key_from_values(dic, val):
    """
    引数valで指定した辞書の値からキーを抽出する（1個のみ）
    参考サイト：https://note.nkmk.me/python-dict-get-key-from-value/
    """
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return None

def make_svr_learning_feature(w_table, s_table, bow_txt_path):
    """
    ユーザが入力した発話から特徴ベクトルを作成する
    ※morphemes -> 分かち書きされた全てのユーザ発話
    
    手順
    1. MeCabを用いて形態素解析 -> ユーザ発話から言語特徴量を抽出する
       言語特徴量としては ... 当該発話に含まれる単語と、WordNetを用いて抽出した同義語からなるbag of wordsの単語特徴ベクトルを用意する
       SVRの学習特徴量としては ... コーパスに含まれる全単語数の次元数のベクトル
                                   入力された発話に含まれる単語に相当する次元を1、含まれない単語に相当する次元を0とする疎ベクトルを作成する
       参考サイト：https://www.pytry3g.com/entry/2018/03/21/181514
    2. 単語極性辞書 -> 発話文に含まれる単語に付与された極性スコアの平均値を算出する
       この2の値を上記の単語特徴ベクトルに付加 -> 1発話の特徴ベクトルとする
    
    bag of wordsの最終的な構造：(row, col) = (発話, 単語および各単語の同義語 + 各発話の極性スコアの平均値)
    """
    # 単語極性辞書の読み込み
    pn_table = {}
    with open(pn_ja_dict_path) as file:
        for line in file:
            line = line.split(":")
            pn_table[line[0]] = float(line[3])
    print("Finished loading pn_ja.dic.")
    
    # コーパス内の全単語に通し番号を割り当てる（0～len(w_table)）
    word2id = {} # {単語: ID}
    for word in w_table.keys(): # word_tableに登録してある各単語
        if word not in word2id.keys():
            word2id[word] = len(word2id)
    
    # bag of wordsを作る
    bow_set = []
    morphemes = get_analyzed_all_usertalks()
    t1 = time.time()
    for wakati_usertalk in morphemes:
        bow = [0] * len(word2id.keys()) # 各発話の全単語分の次元数
        word_pn_scores = 0 # 1発話当たりの極性スコアの記録
        for word in wakati_usertalk.split():
            try:
                if word in pn_table.keys(): # 1発話当たりの極性スコアを加算
                    word_pn_scores += pn_table[word]
                
                if word in word2id.keys(): # 発話内の単語wordが、word_tableに登録されていたら
                    bow[word2id[word]] = 1
                    continue
                
                for synonyms in s_table.value():
                    if word in synonyms: # 発話内の単語wordが、synonym_tableに登録されていたら
                        # wordと同義語の関係を持つ単語のIDを、s_tableで逆算して探す
                        # （wordがsynonymsに含まれている場合に限定しているが、辞書のキーは複数の値を持てないためsynonyms_idは1個のみである）
                        synonyms_id = get_key_from_value(s_table, synonyms)
                        
                        # 該当する単語IDの単語そのものを、w_tableから探す
                        # （1つの単語は複数のsynsetに属する = synonyms_idは複数のw_idsに属する可能性があるので、len(words) >= 1である）
                        words = []
                        for w_ids in w_table.value():
                            if int(synonyms_id) in w_ids: # word_tableとsynonym_tableの単語IDの型を合わせるため、intで変換
                                words.append(get_key_from_value(w_table, w_ids))
                        
                        # bag of wordsを更新する
                        for w in words:
                            bow[word2id[w]] = 1
            except: # 発話内の単語wordが、w_tableに登録されていない場合
                pass
        else:
            # 1発話の単語について全て調べ終わったら、発話文の各単語の極性スコアの平均値を算出する
            # 小数点以下切り捨て防止のため分子にfloat()、0除算防止のため分母に+1
            word_pn_ave = float(word_pn_scores) / (len(wakati_usertalk.split()) + 1)
            bow.append(word_pn_ave)
        bow_set.append(bow)
        
    t2 = time.time()
    print("Finished making bag of words vector: about {} seconds.".format(t2 - t1))
    
    # 作成したbag of wordsをtxtファイルに保存する
    # eval(file.read())で再利用できる
    t3 = time.time()
    with open(bow_txt_path, mode = "w") as file:
        file.write(str(bow_set))
    t4 = time.time()
    print("Finished saving bag of words vector: about {} seconds.".format(t4 - t3))