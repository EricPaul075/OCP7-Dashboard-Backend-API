# *********************************************************************
# Fonctions associées à l'API du projet OC P7
# *********************************************************************
import numpy as np
import pandas as pd
import scipy.stats as sst
import matplotlib.pyplot as plt
import seaborn as sns

import shap

import json
import pickle

import re
import gc
import os

# Variables globales
#cwd = os.getcwd()
#data_path = cwd + '/../data/'
data_path = '../data/'
tmp = data_path + 'tmp/'
if not os.path.exists(tmp):
    os.makedirs(tmp)


def load_id_list():
    """
    Charge la liste des numéros d'identification client.
    :return: list, liste des numéros d'identification.
    """
    with open(data_path+'clients_id_list.txt', "r") as file:
        clients_id_list = json.load(file)
    return clients_id_list

def load_clf():
    """
    Charge le modèle (LightGBM) de prédiction des scores
        client pour l'acceptation de leur demande de prêt.
    :return: LGBMClassifier entrainé.
    """
    with open(data_path + "lgb_buss.pkl", 'rb') as file:
        clf = pickle.load(file)
    return clf

def get_shap_values(Xfile, clf):
    """
    Charge les données et calcule les shap_values.
    :return: shap_values.
    """
    X = pd.read_csv(Xfile, sep=';')
    explainer = shap.Explainer(clf)
    sv = explainer(X)
    return sv

def get_feature_list(fl_wf, fl_abc, fl_curr_app_abc, fl_prev_app_abc,
                     is_wf=True, filter='current'):
    """
    Etablit les liste des features ordonnées selon
        'is_wf' et filtrées selon 'filter'.
    :param fl_wf: list, liste des features
        ordonnées selon les valeurs de Shapley du client.
    :param fl_abc: list, liste des features ordonnées
        alphabétiquement.
    :param fl_prev_app_abc: list, liste des
        features des demandes antérieures de prêt.
    :param fl_curr_app_abc: liste des features
        de la demande de prêt.
    :param is_wf: bool, default=True si l'ordre est selon
        les valeurs de Shapley du client.
    :param filter: list, default='current':
        - 'current': features de la demande de prêt ;
        - 'previous': features des prëts précédents ;
        - 'all': toutes les features.
    :return: list, liste des features raffinées.
    """
    print("fl_prev_app_abc", fl_prev_app_abc[:10])
    print("fl_curr_app_abc", fl_curr_app_abc[:10])
    if filter=='current':
        if is_wf:
            fl = [f for f in fl_wf if f in fl_curr_app_abc]
        else:
            fl = fl_curr_app_abc
    elif filter=='previous':
        if is_wf:
            fl = [f for f in fl_wf if f in fl_prev_app_abc]
        else:
            fl = fl_prev_app_abc
    else:
        fl = fl_wf if is_wf else fl_abc
    print(f"filter={filter}, is_wf={is_wf}, fl: {len(fl)}, {fl[:5]}")
    return fl

def get_cat_num_features_lists():
    """
    Liste des features catégorielles, numériques et
        leur ensemble. Le nom des features est rendu
        compatible avec le modèle LightGBM (regex).
    :return: list, list, list:
        - cat_col: features catégorielles ;
        - num_col: features numériques ;
        - cat_col + num_col.
    """
    # Liste des features catégorielles
    non_input_fl = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    with open(data_path + 'cat_features.txt', "r") as file:
        cat_col = json.load(file)
    cat_col = [re.sub('[^A-Za-z0-9_]+', '', f)
                           for f in cat_col
                           if f not in non_input_fl]
    # Liste des features numériques
    non_input_fl = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    with open(data_path + 'num_features.txt', "r") as file:
        num_col = json.load(file)
    num_col = [re.sub('[^A-Za-z0-9_]+', '', f)
                         for f in num_col
                         if f not in non_input_fl]
    return cat_col, num_col, cat_col + num_col

def bivar_cat_cat(df_cat1_cat_2, alpha=0.05, save=None):
    """
    Effectue l'analyse bivariée entre 2 variables catégorielles.
        Affiche la heatmap et effectue le test du chi2 avec un
        seuil de 5% pour évaluer la dépendance des features.
        Enregistre la représentation graphique dans le fichier
        'save' s'il est spécifié.
    :param df_cat1_cat_2: dataframe, contenant en ligne toutes
        les observations et 2 colonnes, une pour chaque feature.
    :param alpha: float, seuil de test de la pvalue.
    :param save: str, chemin vers le fichier d'enregistrement du
        graphique ; default=None, pas d'enregistrement.
    :return: Rien
    """
    # Format des étiquettes de valeur unique
    df = df_cat1_cat_2.copy()
    features = df.columns.tolist()
    if len(features)!=2:
        return None
    if features[0]==features[1]:
        return None
    for feature in features:
        is_feat_num = True if np.issubdtype(df[feature].dtype, np.number) else False
        if is_feat_num:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            is_int = np.array([x%1==0 for x in pd.unique(df[feature])]).all()
            if is_int:
                df[feature] = df[feature].astype(int)

    # Table de contingence
    cont = df.pivot_table(index=features[0],
                          columns=features[1],
                          aggfunc=len,
                          margins=True,
                          margins_name='total')
    # Table ξ (xi) des corrélations
    tx = cont.loc[:,["total"]]
    ty = cont.loc[["total"],:]
    n = len(df)
    indep = tx.dot(ty) / n
    cont = cont.fillna(0)
    measure = (cont-indep)**2/indep
    xi_n = measure.sum().sum()

    # Test CHI2 (note: xi_n=chi2) - H0: variables indépendantes
    chi2, p_value, ddl, exp = sst.chi2_contingency(cont)
    indep = False if p_value < alpha else True

    # Heatmap (échelle 0-1)
    table = measure/xi_n
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.heatmap(
        table.iloc[:-1,:-1],
        # valeurs de la table des contingences
        annot=cont.iloc[:-1,:-1].astype(int),
        # format de 'annot'
        fmt='d',
        cbar_kws={'label': '← independance    -    dependance →'},
        ax=ax)
    dep = 'variables non corrélées' if indep else 'variables corrélées'
    ax.set_title(f"Heatmap analyse bivariée ({dep})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)

    # Nettoyage des variables
    del cont, tx, ty, n, indep, measure, xi_n, chi2, p_value, ddl, exp, table, dep
    gc.collect()


def bivar_num_num(data, save=None):
    """
    Effectue l'analyse bivariée entre 2 variables numérique.
        Enregistre la représentation graphique dans le fichier
        'save' s'il est spécifié.
    :param data: dataframe, contenant en ligne toutes les
        observations et 2 colonnes, une pour chaque feature.
    :param save: str, chemin vers le fichier d'enregistrement du
        graphique ; default=None, pas d'enregistrement.
    :return: Rien
    """
    pair = data.columns.tolist()
    if len(pair) != 2:
        return None
    if pair[0]==pair[1]:
        return None
    df = data[pair].copy().apply(pd.to_numeric, axis=1)
    coef_p = sst.pearsonr(df[pair[0]], df[pair[1]])[0]

    plt.figure(figsize=(8, 5))
    grid = sns.jointplot(data=df, x=pair[0], y=pair[1], kind="reg", marginal_kws=dict(bins=20, fill=True))
    plt.suptitle(f"Analyse bivariée (corrélation r²={coef_p:.3f})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    del pair, df, coef_p
    gc.collect()


def eta_squared(x, y):
    """
    Calcul du rapport de corrélation entre une variable
        catégorielle x et une variable quantitative y.
    :param x: pandas Series, variable catégorielle.
    :param y: pandas Series, variable numérique.
    :return: float, coefficient de corrélation η²
    """
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj - moyenne_y) ** 2 for yj in y])
    SCE = sum([c['ni'] * (c['moyenne_classe'] - moyenne_y) ** 2 for c in classes])
    eta_squared = SCE / SCT
    del moyenne_y, classes, yi_classe, SCT, SCE
    gc.collect()
    return eta_squared


def welch_ttest(x, y, alpha=0.05):
    """
    Test de Welch avec H0: égalité des moyennes entre x et y.
    :param x: numpy array ou pandas Series
    :param y: numpy array ou pandas Series
    :param alpha:seuil de test de la p-value ; default=0.05
    :return: bool:
        - True: H0 vraie (égalité)
        - False: H0 rejetée (inégalité)
    """
    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
            (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))
    t, p = sst.ttest_ind(x, y, equal_var=False)
    result = p > alpha
    del dof, t, p
    gc.collect()
    return result


from numpy.polynomial import polynomial as P
def bivar_cat_num(df_cat_num, nb_cat=5, alpha=0.05, save=None):
    """
    Effectue l'ANOVA pour une paire de variables (cat, num).
        Enregistre la représentation graphique dans le fichier
        'save' s'il est spécifié.
    :param data: dataframe, contenant en ligne toutes les
        observations et 2 colonnes, une pour chaque feature.
    :param nb_cat: int, nombre maximum de catégories à afficher
        pour la variables catégorielle.
    :param alpha: float, seuil des tests (normalité, Welch, Fligner).
    :param save: str, chemin vers le fichier d'enregistrement du
        graphique ; default=None, pas d'enregistrement.
    :return: Rien
    """
    pair = df_cat_num.columns.tolist()
    if len(pair) != 2:
        return None
    if pair[0] == pair[1]:
        return None
    df = df_cat_num[pair].copy()

    # Format des étiquettes de valeur unique
    cat_feat = df.columns.tolist()[0]
    is_feat_num = True if np.issubdtype(df[cat_feat].dtype, np.number) else False
    if is_feat_num:
        df[cat_feat] = pd.to_numeric(df[cat_feat], errors='coerce')
        is_int = np.array([x % 1 == 0 for x in pd.unique(df[cat_feat])]).all()
        if is_int:
            df[cat_feat] = df[cat_feat].astype(int)

    # Filtrage des catégories qui contiennent moins de 'n_samples_per_cat_min' lignes (min=3)
    df_cat = df.groupby(pair[0], as_index=False).agg(
        means=(pair[1], "mean"),size=(pair[0], "size")).sort_values(
        by='means', ascending=False).reset_index(drop=True)
    n_samples_per_cat_min = 3
    list_cat = df_cat.loc[df_cat['size'] >= n_samples_per_cat_min, pair[0]].tolist()
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.drop(index=df_cat.loc[~df_cat[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.reset_index(drop=True, inplace=True)

    # Filtrage des nb_cat pour lesquelles la moyenne des valeurs numériques est la plus élevée
    df_cat = df_cat.head(nb_cat)
    list_cat = df_cat[pair[0]].head(nb_cat).values.tolist()
    nb_cat = min(nb_cat, len(list_cat))
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df[pair[0]] = pd.Categorical(df[pair[0]], categories=list_cat, ordered=True)

    # Calcul du rapport de corrélation
    eta_sqr = eta_squared(df[pair[0]], df[pair[1]])

    # Remplacement des catégories par une valeur numérique
    df['cat'] = df[pair[0]].copy()
    df['cat'] = df['cat'].astype('object').astype("category")
    df['cat'].replace(df['cat'].cat.categories, [i for i in range(0, len(df['cat'].cat.categories))], inplace=True)
    df['cat'] = df['cat'].astype("int")

    # Tests sur les variables
    # Test de normalité (H0: distribution normale)
    tn = True
    list_norm_neg = {'category': [], 'statistic': [], 'p-value': []}
    for cat in range(nb_cat):
        stat, pvalue = sst.normaltest(df.loc[df['cat'] == cat, pair[1]].values)
        tn = tn and (pvalue > alpha)
        if pvalue <= alpha:
            list_norm_neg['category'].append(cat)
            list_norm_neg['statistic'].append(stat)
            list_norm_neg['p-value'].append(pvalue)

    # Test d'homoscédasticité (H0: variances égales entre les catégories)
    gb = df.groupby(pair[0])[pair[1]]
    stat, p_fligner = sst.fligner(*[gb.get_group(x).values for x in gb.groups.keys()])
    is_fligner_test_positive = p_fligner > alpha

    # Test de Welch (H0: égalité des moyennes entre catégories), si test d'homoscédasticité négatif
    # Table de groupe des catégories en fonction du résultat du test
    tw_true = True
    tw_false = True
    dgr = pd.DataFrame(data=np.arange(len(list_cat)), index=[list_cat], columns=['group'])
    for i in range(len(list_cat) - 1):
        for j in range(i + 1, len(list_cat)):
            is_welch_ttest_positive = welch_ttest(gb.get_group(list_cat[i]).values, gb.get_group(list_cat[j]).values)
            tw_true = tw_true and is_welch_ttest_positive
            tw_false = tw_false and not is_welch_ttest_positive
            # Si le test est positif, les moyennes des 2 catégories sont équivalentes
            if is_welch_ttest_positive:
                gr = dgr.loc[list_cat[i]]['group']
                dgr.at[list_cat[j], 'group'] = gr
    # Valeurs de l'ordonnée pour le grouper les catégories ayant des moyennes non dissemblables
    rows = [-0.5]
    for i in range(1, len(list_cat)):
        if dgr['group'].values[i]!=dgr['group'].values[i-1]:
            rows.append(i-0.5)
    rows.append(len(list_cat)-0.5)

    # Test statistique de Fisher
    dfn = nb_cat - 1
    dfd = df.shape[0] - nb_cat
    F_crit = sst.f.ppf(1 - alpha, dfn, dfd)
    F_stat, p = sst.f_oneway(df['cat'], df[pair[1]])
    sign_F = ">" if F_stat > F_crit else "<"
    sign_p = ">" if p > alpha else "<"
    if (sign_F == ">") and (sign_p == "<"):
        res_test = "positif"
    else:
        res_test = "négatif"

    # Définition des dimensions du graphique global
    fig_h = nb_cat if nb_cat < 6 else int((5 * nb_cat + 40) / 15)

    # Propriétés graphiques
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax = sns.boxplot(
        x=pair[1], y=pair[0], data=df,
        showfliers=False, medianprops=medianprops,
        showmeans=True, meanprops=meanprops, ax=ax)
    xmin, xmax = ax.get_xlim()

    # Tracé des lignes reliant les valeurs moyennes de chaque catégorie
    plt.plot(df_cat.means.values, df_cat.index.values, linestyle='--', c='#000000')

    # Bloc de séparation graphique des groupes de moyennes non différenciées (test de Welch négatif)
    if not tw_true and len(rows)>1:
        for i in range(len(rows)-1):
            plt.fill_between([xmin, xmax], [rows[i], rows[i]], [rows[i+1], rows[i+1]], alpha=0.2)

    # Régression linéaire sur les valeurs moyennes
    reg = P.polyfit(df_cat.means.values, df_cat.index.values, deg=1, full=True)
    yPredict = P.polyval(df_cat.means.values, reg[0])

    # Tracé de la droite de régression linéaire
    plt.plot(df_cat.means.values, yPredict, linewidth=2, linestyle='-', c='#FF0000')

    plt.ylim(top=-1, bottom=nb_cat)
    plt.title(f"ANOVA - analyse bivariée (corrélation η²={eta_sqr:.3f})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)

    # Nettoyage des variables
    del pair, df, df_cat, n_samples_per_cat_min, list_cat
    del nb_cat, eta_sqr, tn, list_norm_neg, cat, stat, pvalue
    del gb, p_fligner, is_fligner_test_positive, tw_true
    del tw_false, dgr, i, j, p, is_welch_ttest_positive
    del rows, dfn, dfd, F_crit, F_stat, sign_F, sign_p
    del res_test, fig_h, medianprops, meanprops, ax
    del xmin, xmax, reg, yPredict, cat_feat, is_feat_num
    if 'gr' in locals(): del gr
    if 'is_int' in locals(): del is_int
    gc.collect()
