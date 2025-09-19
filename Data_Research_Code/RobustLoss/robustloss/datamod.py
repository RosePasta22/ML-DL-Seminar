# datamod.py
import pandas as pd
from sklearn.model_selection import train_test_split
from .schemas import DatasetSchema
from .preprocess import Preprocessor

def prepare_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
):
    assert 0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1
    target = schema.target_name

    df_train, df_temp = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df[target], random_state=random_state
    )
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp, test_size=rel_test,
        stratify=df_temp[target], random_state=random_state
    )

    pp = Preprocessor(schema)
    X_tr, y_tr = pp.fit(df_train)       # train으로만 fit
    X_val, y_val = pp.transform(df_val) # val/test는 transform만
    X_te, y_te = pp.transform(df_test)

    meta = dict(
        n_features=pp.n_features_out_,
        num_classes=pp.num_classes_,
        task_type=pp.task_type_,
        label_encoder=pp.label_encoder
    )
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te), meta