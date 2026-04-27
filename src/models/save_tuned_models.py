"""
Save Tuned Models to Disk
=========================
Run this ONCE after tune_winners.py completes.
Saves every tuned model + its imputer/scaler to models/tuned/
so the Streamlit dashboard can load them without retraining.

Output layout:
  models/tuned/
    {target}.pkl            — fitted sklearn model  (non-MLP)
    {target}_keras.keras    — Keras 3 weights       (MLP)
    {target}_prep.pkl       — {'imputer': ..., 'scaler': ..., 'feature_cols': [...]}

Run:
    python save_tuned_models.py
"""

import json, gc, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.features.leakage_rules import drop_organ_support_leaky_columns

PARAMS_PATH = Path('results/best_hyperparams.json')
DATA_DIR    = Path('data/processed/tournament')
OUT_DIR     = Path('models/tuned')

MATRIX_FILES = {
    'IG':    'X_ig_union.csv',
    'ANOVA': 'X_anova_union.csv',
    'MI':    'X_mi_union.csv',
    'LASSO': 'X_lasso_union.csv',
}
ALL_TARGET_COLS = [
    'mortality','aki_onset','sepsis_onset','ards_onset','liver_injury_onset',
    'need_vent_any','need_vasopressor_any','need_rrt_any',
    'icu_readmit_48h','icu_readmit_7d','los_category','discharge_disposition','los_days'
]
ID_COLS = ['subject_id','hadm_id','stay_id']
TASK_TYPE = {
    'mortality':'binary','aki_onset':'binary','sepsis_onset':'binary',
    'ards_onset':'binary','liver_injury_onset':'binary','need_vent_any':'binary',
    'need_vasopressor_any':'binary','need_rrt_any':'binary',
    'icu_readmit_48h':'binary','icu_readmit_7d':'binary',
    'los_category':'multiclass','discharge_disposition':'multiclass',
}


def load_data(target_name, matrix_name):
    path = DATA_DIR / MATRIX_FILES[matrix_name]
    df   = pd.read_csv(path)
    df.columns = [c.replace('[','').replace(']','').replace('<','lt').replace('>','gt')
                  for c in df.columns]
    df = df.dropna(subset=[target_name]).copy()
    le = LabelEncoder()
    y  = le.fit_transform(df[target_name])
    n_classes   = len(le.classes_)
    drop_cols   = [c for c in ID_COLS + ALL_TARGET_COLS if c in df.columns]
    X           = df.drop(columns=drop_cols)
    X           = drop_organ_support_leaky_columns(X, target_name)
    feat_names  = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    imputer     = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_imp)
    return X_train_sc, y_train, n_classes, feat_names, imputer, scaler


def build_and_train(model_name, params, task_type, n_classes, X_train, y_train):
    if model_name == 'CatBoost':
        p = dict(params, random_seed=42, verbose=0)
        p['loss_function'] = 'Logloss' if task_type=='binary' else 'MultiClass'
        if task_type == 'binary': p['auto_class_weights'] = 'Balanced'
        m = CatBoostClassifier(**p); m.fit(X_train, y_train); return m, False

    elif model_name == 'XGBoost':
        p = dict(params, random_state=42, n_jobs=-1, use_label_encoder=False)
        p['eval_metric'] = 'logloss' if task_type=='binary' else 'mlogloss'
        if task_type != 'binary':
            p.update({'objective':'multi:softprob','num_class':n_classes})
        m = xgb.XGBClassifier(**p); m.fit(X_train, y_train); return m, False

    elif model_name == 'Random Forest':
        m = RandomForestClassifier(**params, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
        m.fit(X_train, y_train); return m, False

    elif model_name == 'Logistic Regression':
        m = LogisticRegression(**params, solver='saga', max_iter=2000,
                               class_weight='balanced', random_state=42)
        m.fit(X_train, y_train); return m, False

    elif model_name == 'Custom MLP':
        u1,u2,u3 = params['units_1'],params['units_2'],params['units_3']
        d1,d2,lr = params['dropout_1'],params['dropout_2'],params['lr']
        bs = params['batch_size']
        val_n = int(0.15*len(X_train))
        Xtr,Xval = X_train[:-val_n], X_train[-val_n:]
        ytr,yval = y_train[:-val_n], y_train[-val_n:]
        model = Sequential([
            Dense(u1,activation='relu',input_shape=(X_train.shape[1],)),
            Dropout(d1), Dense(u2,activation='relu'), Dropout(d2),
            Dense(u3,activation='relu'),
        ])
        if task_type == 'binary':
            model.add(Dense(1,activation='sigmoid'))
            model.compile(optimizer=Adam(lr),loss='binary_crossentropy',metrics=['AUC'])
        else:
            model.add(Dense(n_classes,activation='softmax'))
            model.compile(optimizer=Adam(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        cb = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        model.fit(Xtr,ytr,epochs=100,batch_size=bs,
                  validation_data=(Xval,yval),callbacks=[cb],verbose=0)
        return model, True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    print("="*60)
    print("SAVING TUNED MODELS")
    print("="*60)

    for target_name, cfg in all_params.items():
        model_name  = cfg['model']
        matrix_name = cfg['matrix']
        params      = cfg['params']
        task_type   = TASK_TYPE[target_name]

        print(f"\n[{target_name}]  {model_name} / {matrix_name} matrix")

        X_train, y_train, n_classes, feat_names, imputer, scaler = \
            load_data(target_name, matrix_name)

        model, is_keras = build_and_train(
            model_name, params, task_type, n_classes, X_train, y_train)

        # Save preprocessing
        prep = {'imputer': imputer, 'scaler': scaler, 'feature_cols': feat_names,
                'task_type': task_type, 'n_classes': n_classes,
                'model_name': model_name, 'matrix_name': matrix_name}
        joblib.dump(prep, OUT_DIR / f'{target_name}_prep.pkl')
        print(f"   Saved prep  →  {target_name}_prep.pkl")

        # Save model
        if is_keras:
            keras_path = OUT_DIR / f'{target_name}_keras.keras'
            model.save(str(keras_path))
            print(f"   Saved MLP   →  {keras_path.name}")
            K.clear_session()
        else:
            joblib.dump(model, OUT_DIR / f'{target_name}.pkl')
            print(f"   Saved model →  {target_name}.pkl")

        del model
        gc.collect()

    print("\n[DONE] All models saved to:", OUT_DIR.absolute())
    print("[NEXT] Run: streamlit run app.py")


if __name__ == '__main__':
    main()
