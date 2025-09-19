import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from .schemas import DatasetSchema, TaskType

class Preprocessor:
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.column_transformer: ColumnTransformer | None = None
        self.label_encoder = LabelEncoder()
        self.n_features_out_: int | None = None
        self.num_classes_: int | None = None
        self.task_type_: TaskType | None = None

    def _infer_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if self.schema.numeric_features is None and self.schema.categorical_features is None:
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c != self.schema.target_name]
            cat_cols = [c for c in df.columns 
            if c not in num_cols + [self.schema.target_name]]  
        else:
            num_cols = list(self.schema.numeric_features or [])
            cat_cols = list(self.schema.categorical_features or [])

        # drop_features 적용
        if self.schema.drop_features:
            num_cols = [c for c in num_cols if c not in self.schema.drop_features]
            cat_cols = [c for c in cat_cols if c not in self.schema.drop_features]
        return num_cols, cat_cols
    
    def build_pipeline(self, df: pd.DataFrame):
        num_cols, cat_cols = self._infer_columns(df)
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        # skLearn 버전 호환 (sparse_output vs sparse)
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop"
        )

    def fit(self, df: pd.DataFrame):
        X = df.drop(columns=[self.schema.target_name])
        Y = df[self.schema.target_name].values
        self.build_pipeline(df)
        Xt = self.column_transformer.fit_transform(X)
        Yt = self.label_encoder.fit_transform(Y)
        self.n_features_out_ = Xt.shape[1]
        self.num_classes_ = len(np.unique(Yt))
        # task_type 자동 감지(스키마 미지정 시)
        self.task_type_ = self.schema.task_type or (
            TaskType.BINARY if self.num_classes_ == 2 else TaskType.MULTICLASS
        )
        return Xt, Yt
    
    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.schema.target_name])
        Y = df[self.schema.target_name].values if self.schema.target_name in df else None
        Xt = self.column_transformer.transform(X)
        Yt = self.label_encoder.transform(Y) if Y is not None else None
        return Xt, Yt