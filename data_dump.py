import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb+srv://Meet:md2002dk@cluster0.6tfsldv.mongodb.net/")

DATA_PATH = (r"F:/Insurance_Premium_Prediction/Insurance-Premium-Prediction/insurance.csv")
DATABASE_NAME = "INSURACE"
COLLECTION_NAME = "INSURANCE_COLLECTION"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Dimensions = {df.shape}")

    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    # print(json_record[1])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

