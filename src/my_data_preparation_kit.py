## ※前処理：データセットの整形、データベースの構築
## -> data_preparation1.ipynb内で直接実装および実行している
## ※時間を見つけて、これらもこっちに移行させた方が良いかも

import nltk
import unicodedata
import re
from pathlib import Path
import MeCab
import sqlite3
from pprint import pprint
import json
import time
from collections import OrderedDict

stopwords_txt_path = "../stopwords_jpn.txt"
constructed_corpus_path = "dialogue_corpus.db"
wordnet_path = "../wnjpn.db"
word_table_json_name = "word_table_data.json"
synonym_table_json_name = "synonym_table_data.json"

## ファイル周りの処理
## -> my_feature_vector_creator_kit.pyでも利用する

def convert_dict_to_json(orig_dict, indent = 4):
    """
    辞書からJSON形式に変換する
    参考サイト：https://tmg0525.hatenadiary.jp/entry/2018/03/30/204448
    """
    return json.dumps(orig_dict, indent = indent)

def load_json_as_dict(json_name):
    """
    JSON形式の文字列を辞書として読み込む
    参考サイト：https://note.nkmk.me/python-json-load-dump/
    """
    with open("./" + json_name, "r") as json_file:
        return json.load(json_file, object_pairs_hook = OrderedDict)

def save_dict_as_json(dict_name, json_name, indent = 4):
    """
    （返り値無し）
    word_tableとsynonym_tableをJSONファイルとして保存する
    このノートブックをシャットダウンしたら、JSONファイルの内容を閲覧できるようになる(それまでは何故か内容が更新されていない)
    """
    fw = open(json_name, "w")
    json.dump(dict_name, fw, indent = indent)

## word_tableとsynonym_tableの作成およびJSONファイルへの保存
## 前処理：語の正規化は、ここで利用している

## ※コーパスからユーザ発話のみを取得する関数 get_all_usertalks_from_corpus()
##   -> my_feature_vector_creator_kit.pyでも利用する

def normalize_unicode(text, form = "NFKC"):
    """
    unicodeの正規化
    半角カタカナを全角カタカナに変換したりする
    """
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def normalize_number(text):
    """
    1個以上連続した数字を0で置換
    """
    replaced_text = re.sub(r"\d+", "0", text)
    return replaced_text

def lower_text(text):
    """
    英字を小文字に変換する
    """
    return text.lower()

def normalize(text):
    """
    語の正規化
    """
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text

def get_stopwords():
    """
    ストップワードの取得
    以下のサイトからストップワードの情報を保存する
    http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt
    """
    stopwords_file = Path(stopwords_txt_path)
    with stopwords_file.open("r", encoding="utf-8") as file:
        # 改行文字のみの行は無視
        # 各要素の末尾にある改行文字を除去
        stopwords = [line.rstrip() for line in file if line != "\n"]
    stopwords.remove("時間") # 同じ「時間」でも、「2時間」(接尾語)はあまり重要ではないが「時間が...」は重要となるため
    return stopwords

def judge_necessity_word_stem(node_surface, node_feature):
    """
    語幹を取り出す必要がある単語かどうかを判定する
    条件
    1. 表層形のデータが空ではない
    2. 表層形がget_stopwords()に含まれていない
    3. 単語の品詞が名詞、動詞、形容詞以外(これら以外は解析においてあまり重要ではないため)
    4. 名詞だとしても接尾語ではない(同じ名詞の「時間」でも、「2時間」はあまり重要ではないが「時間が...」は重要)
    """
    surface_has_content = node_surface != ""
    not_in_stopwords = node_surface not in get_stopwords()
    is_meaningful_word = node_feature.split(",")[0] == "名詞" or node_feature.split(",")[0] == "動詞" or node_feature.split(",")[0] == "形容詞"
    not_suffix = node_feature.split(",")[1] != "接尾"
    return surface_has_content and not_in_stopwords and is_meaningful_word and not_suffix

def get_all_usertalks_from_corpus():
    """
    コーパスからユーザ発話のみを取得する
    """
    corpus = sqlite3.connect(constructed_corpus_path)
    cur = corpus.cursor()
    scenes = ["cleaning", "exercise", "game", "lunch", "sleep"]
    dialogue_idx = [str(i) for i in range(200)]
    all_usertalk = []
    
    t1 = time.time()
    # コーパス全体からユーザ発話のみを取得
    for scene in scenes:
        for idx in dialogue_idx:
            dialogue_name = scene + idx
            # 一般的なインジェクション攻撃はテーブル名を変える攻撃が無いから、?で置き換えるやつが実装されていない（かも）
            # テーブル名を変える攻撃の場合は文字列の長さで防げるので、formatで大丈夫
            # "select * from {}".format(table)にすると、tableをA where '1'='1'にすれば攻撃できるけど
            # 存在するテーブル名の長さよりも長ければ攻撃だと判定できる（len(table) > (存在するテーブル名): 攻撃だと判定）
            # by.師匠
            cur.execute("select * from {}".format(dialogue_name))
            all_usertalk.append([talk[1] for i, talk in enumerate(cur.fetchall()) if i % 2 == 1])
    t2 = time.time()
    print("Finished getting all user talks from dialogue_corpus.db: about {} seconds.".format(t2 - t1))
    return all_usertalk

def make_and_save_tables():
    """
    2つの辞書を作ってJSONファイルとして保存する
    データの形式
    word_table ... {word -> str: wordid -> list}
    synonym_table ... {wordid[i] -> init: synonym -> list}
    """
    all_usertalk = get_all_usertalks_from_corpus()
    wn_db = sqlite3.connect(wordnet_path)
    wn_c = wn_db.cursor()
    m = MeCab.Tagger("-Ochasen")
    word_table = {}
    synonym_table = {}
    
    t3 = time.time()
    for i, usertalks in enumerate(all_usertalk): # usertalks：1対話それぞれにおけるユーザ発話を格納しているリスト
        for usertalk in usertalks:
            ## 単語IDの取得
            # 発話の正規化
            usertalk = normalize(usertalk)
            # MeCabによる形態素解析で、発話に含まれる単語を抽出する
            # node.surface -> 表層形(区切った単語自体)
            # node.feature -> 表層形の詳細(品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音)
            node = m.parseToNode(usertalk)
            while node:
                if judge_necessity_word_stem(node.surface, node.feature):
                    # 各単語の語幹を取り出す
                    orig_form = node.feature.split(",")[6]
                    # 抽出した単語に対応する単語IDを取得する
                    wn_c.execute("select * from word where lemma = ?", (orig_form,))
                    key = node.surface
                    value = [i[0] for i in wn_c.fetchall()] # i -> [(単語ID, jpnかeng, 単語, None, 品詞), (...)] -> 単語IDのみ取得
                    word_table[key] = value
                node = node.next

            ## 各単語の同義語の取得
            for word, wordids in word_table.items():
                if wordids != []:
                    for wordid in wordids:
                        # wordidに対応する単語が属するsynsetをsenseテーブルから取得する
                        # synset -> 同義語をまとめてある単語の集まり
                        # 注：1つの単語は複数のsynsetに属する
                        wn_c.execute("select * from sense where wordid = ?", (wordid,))
                        synsets = [s[0] for s in wn_c.fetchall()] # s -> [(synset, wordid, lang, rank, lexid, freq, src), (...)] -> synsetのみを取得
                        
                        # 1つのwordidが持つ同義語全てを格納するリスト
                        synonyms_of_wordid = []
                        for synset in synsets:
                            # synset内の単語をWordNetのwordテーブルから探す
                            wn_c.execute("select lemma from sense, word where synset = ? and word.lang = \"jpn\" and sense.wordid = word.wordid", (synset,))
                            for s in wn_c.fetchall(): # s -> [("同義語",), ("同義語",), ...] -> 同義語の文字列のみ取得
                                if s[0] not in synonyms_of_wordid:
                                    synonyms_of_wordid.append(s[0])
                        key = wordid
                        value = synonyms_of_wordid
                        synonym_table[key] = value
        print("Finished registering {} usertalks.".format(i + 1))
    t4 = time.time()
    print("Finished registering all wordids to word_table.")
    print("Finished registering all synonyms to synonym_table.")
    
    t5 = time.time()
    save_dict_as_json(word_table, word_table_json_name, 4)
    print("Finished registering a dict: word_table with file: {}".format(word_table_json_name))
    save_dict_as_json(synonym_table, synonym_table_json_name, 4)
    print("Finished registering a dict: synonym_table with file: {}".format(synonym_table_json_name))
    t6 = time.time()
    
    print("Finished making word_table, synonym_table took about {} seconds.".format(t4 - t3))
    print("Finished saving word_table, synonym_table took about {} seconds.".format(t6 - t5))