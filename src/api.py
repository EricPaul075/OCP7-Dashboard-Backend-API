from api_functions import *
from fastapi import FastAPI
from starlette.responses import FileResponse
import shap

from typing import Union, Optional
from pydantic import BaseModel, Field

# Chargement des données (fixes) de modèle, shap_values et listes de features
print('Chargements et initialisations...')
clf = load_clf()
shap_values = get_shap_values(Xfile=data_path+"P7_dashboard_Xdata_10pc.csv", clf=clf)
clients_id_list = load_id_list()
features_list = shap_values.feature_names
fl_abc = sorted(features_list)
prefixes = ('BURO_', 'ACTIVE_', 'CLOSED_', 'PREV_', 'APPROVED_', 'REFUSED_', 'POS_', 'INSTAL_', 'CC_')
fl_prev_app_abc = [f for f in fl_abc if f.startswith(prefixes)]
fl_curr_app_abc = [f for f in fl_abc if not f.startswith(prefixes)]
cat_col, num_col, columns = get_cat_num_features_lists()
class_value = 1
print('\t→ chargements et initialisations terminés')

class GDO_client_list(BaseModel):
    id_list: list  # liste des identifiants client

class GDO_features(BaseModel):
    all: list  # Liste de l'ensemble des features
    cat: list  # Liste des features catégorielles
    num: list  # Liste des features numériques

class CDO_score(BaseModel):
    client_id : int  # N° id du client
    score: float  # score de prédiction

class DO_feature_selection(BaseModel):
    client_id : int  # N° id du client
    is_wf : bool = True  # si la liste suit l'ordre de l'impact local
    filter: str = 'current'  # filtre ['all', 'current', 'previous']
    feature_selection: list  # liste pour le menu de sélection


# Application FastAPI
app = FastAPI(title='OC P7 Dashboard API', version='1.0',
              description="API pour le dashboard du projet OC P7 réalisé avec FastAPI")

# Root
@app.get('/')
def root():
    """
    Message de connection à la racine du serveur.
    :return: dict, message de fonctionnement de l'API
    """
    return {"message": "API en cours d'exécution"}

# Home
@app.get('/home')
def home():
    """
    Message d'identification de l'API.
    :return: dict: message explicitant l'API.
    """
    return {"message": "API pour le dashboard d'OC P7"}

# Liste des id des clients
@app.get("/clients_list", response_model=GDO_client_list)
def get_clients_list():
    """
    Liste des n° d'identification des clients.
    :return: GDO_client_list (liste)
    """
    return GDO_client_list(id_list=clients_id_list)

# Liste des features
@app.get("/feature_lists", response_model=GDO_features)
def get_feature_lists():
    """
    Listes des features du modèle:
        - all: toutes les features
        - cat: features catégorielles
        - num: features numériques
    :return: GDO_features (listes)
    """
    return GDO_features(all=features_list, cat=cat_col, num=num_col)

# Graphe impact global des features
@app.get('/global_impact', response_class=FileResponse)
async def get_global_impact(max_feat: int = 20):
    """
    Graphe d'impact global des features.
    :param max_feat: int, nombre de features à représenter
        sur le graphique.
    :return: FileResponse (fichier PNG du graphe)
    """
    filepath = tmp + f"gfgi_{max_feat}.png"
    if not os.path.exists(filepath):
        plt.subplots(figsize=(10, 20))
        shap.summary_plot(shap_values[..., class_value], shap_values.data, max_display=max_feat, show=False)
        plt.tight_layout()
        plt.savefig(filepath)
    return FileResponse(filepath)

@app.get("/graph_bivar", response_class=FileResponse)
async def graph_bivar(feature_1: str, feature_2: str):
    """
    Graphe d'analyse bivariée.
    :param feature_1: str, feature
    :param feature_2: str, feature
    :return: FileResponse (fichier PNG du graphe)
    """
    # Vérifie les features et nom du fichier graphique s'il existe
    if feature_1 in features_list and feature_2 in features_list:
        f1_idx = features_list.index(feature_1)
        f2_idx = features_list.index(feature_2)
    else:
        return FileResponse("")
    filepath_1 = tmp + f"bivar{f1_idx}_{f2_idx}.png"
    filepath_2 = tmp + f"bivar{f2_idx}_{f1_idx}.png"

    # Vérification si le fichier existe + nom du fichier image de sortie
    if os.path.exists(filepath_1):
        file_exists = True
        filepath = filepath_1
    elif os.path.exists(filepath_2):
        file_exists = True
        filepath = filepath_2
    else:
        file_exists = False
        filepath = tmp + f"bivar{f1_idx}_{f2_idx}.png"

    # Taille de l'image + fichier graphe s'il n'existe pas déjà
    if feature_1 in cat_col and feature_2 in cat_col:
        if not file_exists:
            df_biv = pd.DataFrame(shap_values[:, [feature_1, feature_2], class_value].data, columns=[feature_1, feature_2])
            bivar_cat_cat(df_biv, save=filepath)
    elif feature_1 in num_col and feature_2 in num_col:
        if not file_exists:
            df_biv = pd.DataFrame(shap_values[:, [feature_1, feature_2], class_value].data, columns=[feature_1, feature_2])
            bivar_num_num(df_biv, save=filepath)
    elif feature_1 in cat_col:
        if not file_exists:
            df_biv = pd.DataFrame(shap_values[:, [feature_1, feature_2], class_value].data, columns=[feature_1, feature_2])
            bivar_cat_num(df_biv, save=filepath)
    else:
        if not file_exists:
            df_biv = pd.DataFrame(shap_values[:, [feature_2, feature_1], class_value].data, columns=[feature_2, feature_1])
            bivar_cat_num(df_biv, save=filepath)
    return FileResponse(filepath)

# Mise à jour de 'item' avec le n° de client
@app.get("/{client_id}", response_model=CDO_score)
def get_client_score(client_id: int):
    """
    Score de prédiction pour le prêt du client.
    :param client_id: int, n° d'identification du client.
    :return: CDO_score
    """
    sample = clients_id_list.index(client_id)
    score = clf.predict_proba(shap_values.data[sample].copy().reshape(1, -1))[:, 1][0]
    return CDO_score(client_id=client_id, score=score)

# Graphe impact local des features
@app.get('/{client_id}/local_impact', response_class=FileResponse)
async def local_impact(client_id: int, max_feat: int = 16):
    """
    Graphe d'impact local des features
    :param client_id: int, n° d'identification du client.
    :param max_feat: int, nombre de features à représenter
        sur le graphique.
    :return: FileResponse (fichier PNG du graphe)
    """
    filepath = tmp + f"gfli_{client_id}_{max_feat}.png"
    if not os.path.exists(filepath):
        plt.subplots(figsize=(10, 20))
        sample = clients_id_list.index(client_id)
        shap.plots.waterfall(shap_values[sample][:, class_value], max_display=max_feat, show=False)
        plt.tight_layout()
        plt.savefig(filepath)
    return FileResponse(filepath)

# Liste des features
@app.get('/{client_id}/feature_selection', response_model=DO_feature_selection)
def get_features_selection(client_id: int, is_wf: bool = True, filter: str = 'current'):
    """
    Génère la liste de sélection des features.
    :param client_id: int, n° d'identification du client.
    :param is_wf: bool, default=True, si la liste est à
        ordonner selon l'impact local (décroissant). Si
        False, la liste est par ordre alphabétique.
    :param filter: list, filtre éventuellement la liste:
        - 'all': toutes les features ;
        - 'current': features de la demande de prêt ;
        - 'previous': features des prêts antérieurs.
    :return: DO_feature_selection (liste)
    """
    if is_wf == True and client_id is not None:
        sample = clients_id_list.index(client_id)
        df_local_shap_val = pd.DataFrame(shap_values[..., class_value].values[sample],
                                         index=shap_values.feature_names, columns=['Shap_val'])
        df_local_shap_val = df_local_shap_val.abs().sort_values(by='Shap_val', ascending=False)
        fs_wf = df_local_shap_val.index.values.tolist()
    else:
        fs_wf = None
    fs = get_feature_list(fs_wf, fl_abc, fl_curr_app_abc,
                          fl_prev_app_abc, is_wf=is_wf, filter=filter)
    return DO_feature_selection(client_id=client_id, is_wf=is_wf,
                                filter=filter, feature_selection=fs)

@app.get('/{client_id}/feature', response_class=FileResponse)
async def graph_feature(client_id: int, feature: str):
    """
    Graphe de la feature avec le positionnement du client.
    :param client_id: int, n° d'identification du client.
    :param feature: str, nom de la feature.
    :return: FileResponse (fichier PNG du graphe)
    """
    f_idx = features_list.index(feature)
    filepath = tmp + f"feature_{client_id}_{f_idx}.png"
    if not os.path.exists(filepath):
        sample = clients_id_list.index(client_id)
        feature_iloc = shap_values.feature_names.index(feature)
        feature_value = shap_values.data[sample, feature_iloc]
        feature_shap_value = shap_values[sample, feature, class_value].values
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.scatter(shap_values[:, feature, class_value], color=shap_values[..., class_value],
                           ax=ax, show=False)
        ax.plot([feature_value, feature_value], [ax.get_ylim()[0], feature_shap_value],
                ls='--', lw=1, color='springgreen')
        ax.plot([ax.get_xlim()[0], feature_value], [feature_shap_value, feature_shap_value],
                ls='--', lw=1, color='springgreen')
        ax.plot(feature_value, feature_shap_value, marker='s', color='springgreen')
        ax.set_title(f"Feature {feature}", fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath)
    return FileResponse(filepath)


# Lancement de l'API par exécution du fichier python
import uvicorn
if __name__ == '__main__':
    uvicorn.run("api:app", reload=True)  # pour fonctionnement par défaut
    #uvicorn.run("api:app", host='0.0.0.0', port=8000, reload=False)  # pour fonctionnement dans conteneur
