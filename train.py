import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor
import bentoml

def train_xgb():

    data = pd.read_csv('./data/listingss.csv')

    columns = [ 'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
        'minimum_nights', 'number_of_reviews', 
        'calculated_host_listings_count',
        'availability_365', 'number_of_reviews_ltm']

    data = data[columns]
    data = data.drop_duplicates().reset_index(drop=True)

    data["price"] = np.log1p(data["price"] )

    df_full_train, df_test = train_test_split(data, test_size = 0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train['price']
    y_val = df_val['price']
    y_test = df_test['price']

    del df_train['price']
    del df_val['price']
    del df_test['price']

    df_train_dict = df_train.to_dict(orient='records')
    df_val_dict = df_val.to_dict(orient='records')
    df_test_dict = df_test.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(df_train_dict)
    X_val = dv.transform(df_val_dict)
    X_test = dv.transform(df_test_dict)

    xgb = XGBRegressor(max_depth=6)

    xgb.fit(X_train, y_train)    

    saved_model = bentoml.xgboost.save_model(
    name='price_prediction_model',
    model=xgb,
    custom_objects={
        "preprocessor": dv
    },
    signatures={
        "predict":{
            "batchable": True,
            "batch_dim": 0
        }
    }
    )

    print(f"Model saved: {saved_model}")

if __name__ == "__main__":
    train_xgb()
