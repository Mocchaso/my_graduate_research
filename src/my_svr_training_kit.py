import numpy as np
import scipy.sparse as sci_sp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.externals import joblib
import time
from my_data_preparation_kit import get_all_usertalk_data_from_corpus
from my_feature_vector_creator_kit import get_all_words_in_corpus

constructed_corpus_path = "dialogue_corpus.db"
svr_bow_path = "svr_bow_vec.npz"
# 学習済みモデルの保存先
trained_ridge_path = "trained_ridge_model.pkl"
trained_linear_svr_path = "trained_linear_svr_model.pkl"
trained_rbf_svr_path = "trained_rbf_svr_model.pkl"
# パラメータのチューニングを終えたモデルの保存先
tuned_ridge_path = "tuned_ridge_model.pkl"
tuned_linear_svr_path = "tuned_linear_svr_model.pkl"
tuned_rbf_svr_path = "tuned_rbf_svr_model.pkl"

def gen_cv(y):
    """
    交差検証データのジェネレータ
    -> 学習データの1/4を交差検証データにする(学習データ:テストデータ = 8:2)
    -> 学習データ:交差検証データ:テストデータ = 6:2:2
    
    参考サイト：https://qiita.com/koshian2/items/baa51826147c3d538652
    """
    m_train = np.floor(len(y) * 0.75).astype(int) # このキャストをintにしないと後にハマる
    train_indices = np.arange(m_train) # 0番目～m_train番目までのインデックス
    test_indices = np.arange(m_train, len(y)) # m_train + 1番目～len(y) - 1番目
    yield (train_indices, test_indices)

def prepare_traindata():
    """
    ユーザの受諾度合いの推定で使うSVRの学習データを準備する
    学習データ ... 入力：ユーザ発話の特徴ベクトル、出力：ユーザの受諾度合いの正解ラベル
    正解ラベル -> 発話に付与される5段階のユーザの受諾度合いラベル
    
    参考サイト：
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
    https://web-salad.hateblo.jp/entry/2014/11/09/090000
    """
    # 疎行列として保存したベクトルを読み込む(csr_matrixで保存している)
    # 参考サイト：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz
    t1 = time.time()
    X = sci_sp.load_npz(svr_bow_path)
    t2 = time.time()
    print("Finished loading input vector, bag of words: about {} seconds.".format(t2 - t1))
    print("shape of bag of words vector: {}".format(X.shape)) # (9754, 6548)のはず
    
    # 出力となる受諾度合いラベルのベクトルを作成する
    t3 = time.time()
    all_usertalk_data = get_all_usertalk_data_from_corpus()
    y = [] # SVRの出力：ユーザの受諾度合いの正解ラベル [acceptance]
    for usertalk_data in all_usertalk_data:
        if usertalk_data[3] != "NONE": # NONEとアノテートされている受諾度合いラベルは、学習データから除く
            an_acceptance_label = int(usertalk_data[3])
            y.append(an_acceptance_label)
    t4 = time.time()
    print("Finished making output vector, user acceptance labels: about {} seconds.".format(t4 - t3))
    print("len(y) = {}".format(len(y))) # 9754のはず
    
    # 訓練データとテストデータの分割(8:2)
    t5 = time.time()
    # 疎行列は(X-平均)/標準偏差の標準化ができないっぽいので、np.ndarrayに変換
    X = X.toarray()
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    t6 = time.time()
    print("訓練データ、交差検証データ、テストデータの数 = ", end="")
    print(len(next(gen_cv(y_train))[0]), len(next(gen_cv(y_train))[1]), len(y_test))
    print("Finished splitting data: about {} seconds.".format(t6 - t5))
    print()
    
    # 訓練データを基準に標準化(平均、標準偏差で標準化)
    t7 = time.time()
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    # テストデータも標準化
    X_test_norm = scaler.transform(X_test)
    t8 = time.time()
    print("Finished standardization train_data and test_data: about {} seconds.".format(t8 - t7))
    
    # 予測モデルをシリアライズ（訓練済みのモデルをファイルに保存）
    #t9 = time.time()
    #joblib.dump(clf1, trained_svr_path)
    #t10 = time.time()
    #print("Finished saving SVR model as {}: about {} seconds.".format(trained_svr_path, t10 - t9))
    
    print("Finished preparing train_data of SVR.")
    print()
    return X_train_norm, X_test_norm, y_train, y_test

def train_svr(X_train_norm, X_test_norm, y_train, y_test):
    """
    学習データ:テストデータ = 8:2
    -> ホールドアウト法に基づき、学習させた後にテストデータでスコアを計算する
    """
    print("リッジ回帰")
    clf1 = Ridge(alpha=1.0)
    t1 = time.time()
    clf1.fit(X_train_norm, y_train)
    t2 = time.time()
    print("Finished fitting Ridge model: about {} seconds.".format(t2 - t1))
    t3 = time.time()
    score1 = clf1.score(X_test_norm, y_test)
    t4 = time.time()
    print("accuracy of Ridge model: {}, about {} seconds.".format(score1, t4 - t3))
    # 予測モデルをシリアライズ（訓練済みのモデルをファイルに保存）
    t5 = time.time()
    joblib.dump(clf1, trained_ridge_path)
    t6 = time.time()
    print("Finished saving Ridge model as {}: about {} seconds.".format(trained_ridge_path, t6 - t5))
    print()

    print("カーネル無し(線形カーネル)")
    clf2 = SVR(kernel="linear")
    t7 = time.time()
    clf2.fit(X_train_norm, y_train)
    t8 = time.time()
    print("Finished fitting Linear SVR model: about {} seconds.".format(t8 - t7))
    t9 = time.time()
    score2 = clf2.score(X_test_norm, y_test)
    t10 = time.time()
    print("accuracy of Linear SVR model: {}, about {} seconds.".format(score2, t10 - t9))
    t11 = time.time()
    joblib.dump(clf2, trained_linear_svr_path)
    t12 = time.time()
    print("Finished saving Linear SVR model as {}: about {} seconds.".format(trained_linear_svr_path, t12 - t11))
    print()

    print("ガウシアンカーネル")
    clf3 = SVR(kernel="rbf")
    t13 = time.time()
    clf3.fit(X_train_norm, y_train)
    t14 = time.time()
    print("Finished fitting RBF SVR model: about {} seconds.".format(t14 - t13))
    t15 = time.time()
    score3 = clf3.score(X_test_norm, y_test)
    t16 = time.time()
    print("accuracy of RBF SVR model: {}, about {} seconds.".format(score3, t16 - t15))
    t17 = time.time()
    joblib.dump(clf3, trained_rbf_svr_path)
    t18 = time.time()
    print("Finished saving RBF SVR model as {}: about {} seconds.".format(trained_rbf_svr_path, t18 - t17))
    print()

    print("Finished training and caluculating scores of 3 models, and saving trained_models.")
    print()

## チューニングの処理をまとめた関数
## モデルの学習の結果、ガウシアンカーネルのチューニングをするのが一番効果的っぽい。

def tuning_ridge(X_train_norm, X_test_norm, y_train, y_test):
    """
    リッジ回帰のパラメータチューニング
    """
    print("リッジ回帰")
    t1 = time.time()
    params_cnt = 2 # パラメータの候補値の個数
    params = {"alpha": np.logspace(-2, 4, params_cnt)}
    gridsearch = GridSearchCV(Ridge(), params, cv = gen_cv(y_train), scoring = "r2", return_train_score = True)
    gridsearch.fit(X_train_norm, y_train)
    t2 = time.time()
    print("Finished tuning parameters: about {} seconds.".format(t2 - t1))
    print("αのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_, "精度 =", gridsearch.best_score_)
    print()
    # チューニングしたαでフィット
    t3 = time.time()
    regr = Ridge(alpha=gridsearch.best_params_["alpha"])
    train_indices = next(gen_cv(y_train))[0]
    valid_indices = next(gen_cv(y_train))[1]
    regr.fit(X_train_norm[train_indices, :], y_train[train_indices])
    t4 = time.time()
    print("Finished fitting with tuned parameters: about {} seconds.".format(t4 - t3))
    print("切片と係数")
    print(regr.intercept_)
    print(regr.coef_)
    print()
    # テストデータの精度を計算
    t5 = time.time()
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(X_test_norm, y_test))
    print()
    print("※参考")
    print("訓練データの精度 =", regr.score(X_train_norm[train_indices, :], y_train[train_indices]))
    print("交差検証データの精度 =", regr.score(X_train_norm[valid_indices, :], y_train[valid_indices]))
    t6 = time.time()
    print("Finished calculating accuracies: about {} seconds.".format(t6 - t5))
    print()
    # チューニングしたモデルの保存
    t7 = time.time()
    joblib.dump(regr, tuned_ridge_path)
    t8 = time.time()
    print("Finished saving Ridge model as {}: about {} seconds.".format(tuned_ridge_path, t8 - t7))
    print()

def tuning_no_kernel_svr(X_train_norm, X_test_norm, y_train, y_test):
    """
    カーネル無し(線形カーネル)のパラメータチューニング
    """
    print("カーネル無し(線形カーネル)")
    t1 = time.time()
    params_cnt = 2 # パラメータの候補値の個数
    params = {"C": np.logspace(0, 1, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
    gridsearch = GridSearchCV(SVR(kernel = "linear"), params, cv = gen_cv(y_train), scoring = "r2", return_train_score = True)
    gridsearch.fit(X_train_norm, y_train)
    t2 = time.time()
    print("Finished tuning parameters: about {} seconds.".format(t2 - t1))
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()
    # チューニングしたハイパーパラメーターをフィット
    t3 = time.time()
    regr = SVR(kernel = "linear", C = gridsearch.best_params_["C"], epsilon = gridsearch.best_params_["epsilon"])
    train_indices = next(gen_cv(y_train))[0]
    valid_indices = next(gen_cv(y_train))[1]
    regr.fit(X_train_norm[train_indices, :], y_train[train_indices])
    t4 = time.time()
    print("Finished fitting with tuned parameters: about {} seconds.".format(t4 - t3))
    print("切片と係数")
    print(regr.intercept_)
    print(regr.coef_)
    print()
    # テストデータの精度を計算
    t5 = time.time()
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(X_test_norm, y_test))
    print()
    print("※参考")
    print("訓練データの精度 =", regr.score(X_train_norm[train_indices, :], y_train[train_indices]))
    print("交差検証データの精度 =", regr.score(X_train_norm[valid_indices, :], y_train[valid_indices]))
    t6 = time.time()
    print("Finished calculating accuracies: about {} seconds.".format(t6 - t5))
    print()
    # チューニングしたモデルの保存
    t7 = time.time()
    joblib.dump(regr, tuned_linear_svr_path)
    t8 = time.time()
    print("Finished saving Linear SVR model as {}: about {} seconds.".format(tuned_linear_svr_path, t8 - t7))
    print()

def tuning_rbf_kernel_svr(X_train_norm, X_test_norm, y_train, y_test):
    print("ガウシアンカーネル")
    t1 = time.time()
    params_cnt = 20 # パラメータの候補値の個数
    params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
    gridsearch = GridSearchCV(SVR(), params, cv = gen_cv(y_train), scoring = "r2", return_train_score = True)
    gridsearch.fit(X_train_norm, y_train)
    t2 = time.time()
    print("Finished tuning parameters: about {} seconds.".format(t2 - t1))
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()
    # チューニングしたC,εでフィット
    t3 = time.time()
    regr = SVR(C = gridsearch.best_params_["C"], epsilon = gridsearch.best_params_["epsilon"])
    train_indices = next(gen_cv(y_train))[0]
    valid_indices = next(gen_cv(y_train))[1]
    regr.fit(X_train_norm[train_indices, :], y_train[train_indices])
    t4 = time.time()
    print("Finished fitting with tuned parameters: about {} seconds.".format(t4 - t3))
    print()
    # テストデータの精度を計算
    t5 = time.time()
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(X_test_norm, y_test))
    print()
    print("※参考")
    print("訓練データの精度 =", regr.score(X_train_norm[train_indices, :], y_train[train_indices]))
    print("交差検証データの精度 =", regr.score(X_train_norm[valid_indices, :], y_train[valid_indices]))
    t6 = time.time()
    print("Finished calculating accuracies: about {} seconds.".format(t6 - t5))
    print()
    # チューニングしたモデルの保存
    t7 = time.time()
    joblib.dump(regr, tuned_rbf_svr_path)
    t8 = time.time()
    print("Finished saving RBF SVR model as {}: about {} seconds.".format(tuned_rbf_svr_path, t8 - t7))
    print()
