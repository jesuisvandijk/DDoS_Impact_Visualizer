import dataset
from time import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

from article_to_event_level import embed_articles, add_entities_to_df, DATA_FILE

#-------------------------------------------------------------------------------#
#-----------------------------SETTINGS------------------------------------------#
#-------------------------------------------------------------------------------#

DATA_FILE = DATA_FILE
FULL_DATA_FILE = 'Data/07-full-without-annotated-alerts.json'

content_col = "Content"
RANDOM_STATE = 25

# Set to True to predict labels on the full unannotated dataset. Takes time
run_on_full_df = False

#-------------------------------------------------------------------------------#
#-----------------------------FUNCTIONS-----------------------------------------#
#-------------------------------------------------------------------------------#

#Gets the relevant data. Filters on News only by default. Returns DatFrame and sentence embeddings
def get_data(filepath, news_only=False):
    df = dataset.get_df(filepath)
    if news_only:
        df = df[df['Alert Type'] == 'News'].reset_index(drop=True)

    if 'relevant' in df.columns:
        before = len(df)
        df = df[df['relevant'] == True].reset_index(drop=True)
        print(f"  Dropped {before - len(df)} irrelevant (non-cyber) articles.")

    pestle_dims = ['Political', 'Economic', 'Social', 'Technological']
    before = len(df)
    df = df.dropna(subset=pestle_dims).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} articles with missing PESTLE scores.")

    # Ensure scores are integers, not floats
    for dim in pestle_dims:
        df[dim] = df[dim].astype(int)
    features = embed_articles(add_entities_to_df(df,content_col),content_col)
    return df, features

def build_best_pipeline():
    return make_pipeline(

        StandardScaler(),
#Base: k=rbf, c=1, e=0.1
        SVR(
            kernel='rbf',
            C=10,
            epsilon=0.2
        )
    )


def create_confusion_plot(features, labels, dimension_name):
    pipeline = build_best_pipeline()
    pipeline_name = '[' + content_col + '] ' + " ".join([name for name, _ in pipeline.steps])
    print('\n' + pipeline_name)
    last_alg_name = pipeline.steps[-1][0]

    skf = StratifiedKFold(n_splits=3, random_state=56, shuffle=True)
    test_lab = []
    pred_lab = []
    maes = []

    count = 1
    for train_index, test_index in skf.split(features, labels):
        print('Processing fold:', count)

        features_train = pd.DataFrame(features).iloc[train_index]
        features_test = pd.DataFrame(features).iloc[test_index]

        labels_train = pd.DataFrame(labels).iloc[train_index].values.ravel()
        labels_test = pd.DataFrame(labels).iloc[test_index].values.ravel()


        sample_weights = compute_sample_weight('balanced', labels_train)
        pipeline.fit(features_train, labels_train, svr__sample_weight=sample_weights)
        y_pred = np.clip(np.round(pipeline.predict(features_test)), 0, 3).astype(int)

        mae = mean_absolute_error(labels_test, y_pred)
        maes.append(mae)

        print(f"MAE fold {count}: {mae:.3f}")

        test_lab.append(labels_test)
        pred_lab.append(y_pred)
        count += 1

    test_lab_final = np.concatenate(test_lab)
    pred_lab_final = np.concatenate(pred_lab)

    rmse = np.sqrt(mean_squared_error(test_lab_final, pred_lab_final))
    r2 = r2_score(test_lab_final, pred_lab_final)
    spearman, _ = spearmanr(test_lab_final, pred_lab_final)

    score_levels = sorted(np.unique(test_lab_final))

    print('\nFULL REPORT')
    print(f"  MAE      : {np.mean(maes):.3f}  (std: {np.std(maes):.3f})")
    print(f"  RMSE     : {rmse:.3f}")
    print(f"  Spearman : {spearman:.3f}")

    conf_mat = confusion_matrix(test_lab_final, pred_lab_final, labels=score_levels)
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=[f'Pred {s}' for s in score_levels],
                yticklabels=[f'True {s}' for s in score_levels])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(str(last_alg_name +" " + dimension_name))
    plt.tight_layout()
    plt.savefig(f'Dashboard/Outputfiles/confusion_{dimension_name}.pdf', format='pdf')
#    plt.show()
    plt.close()


def predict_values(features, labels, dimension_name, df_full, features_full):

    pipeline = build_best_pipeline()
    sample_weights = compute_sample_weight('balanced', labels)
    pipeline.fit(features, labels, svr__sample_weight=sample_weights)

    y_pred = np.clip(np.round(pipeline.predict(features_full)), 0, 3).astype(int)

    # Save predictions to disk
    df_full[f'predicted_{dimension_name}'] = y_pred
    dataset.writedatasetodisk(df_full, f'predicted_{dimension_name}.json')
    print(f"  Predictions saved to predicted_{dimension_name}.json")
    print(f"  Predicted score distribution:\n{df_full[f'predicted_{dimension_name}'].value_counts().sort_index()}")



def main():
    df, features = get_data(DATA_FILE)

    if run_on_full_df:
        print("\nLoading and embedding full dataset (once)…")
        df_full = dataset.get_df(FULL_DATA_FILE)
        print(f"  Loaded {len(df_full)} articles from {FULL_DATA_FILE}.")
        features_full = embed_articles(df_full, content_col)
    else:
        df_full = None
        features_full = None

    for dim in ['Political', 'Economic', 'Social', 'Technological']:
        print(f"\n===== Training dimension: {dim} =====")
        labels = df[dim]

        print("\nStep 1: Evaluating with cross-validation…")
        create_confusion_plot(features, labels, dim)

        if run_on_full_df:
            print("\nStep 2: Predicting on full dataset...")
            predict_values(features, labels, dim, df_full, features_full)
        else:
            print("\nStep 2: Skipping full dataset prediction (RUN_ON_FULL_DF = False).")

    print("\nDone!")

main()