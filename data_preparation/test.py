import os
import pickle
import json

if __name__ == "__main__":
    FineDivingPath = "FineDiving.json"
    FineDivingDataPaths = json.load(open(FineDivingPath, 'r'))
    print(FineDivingDataPaths)

    # load DataSplit
    DataSetPaths = FineDivingDataPaths['dataSet']
    train_split = pickle.load(open(DataSetPaths['train_split'], 'rb'))
    test_split = pickle.load(open(DataSetPaths['test_split'], 'rb'))

    # Load Sub_action_Types
    SubActionPath = DataSetPaths['sub_action']
    SubActionTypes = pickle.load(open(SubActionPath, 'rb'))
    print(SubActionTypes)

    SubActionTypeResultPath = DataSetPaths['sub_action_result']
    SubActionTypeResult = pickle.load(open(SubActionTypeResultPath, 'rb'))
    print(SubActionTypeResult)

    for key, value in SubActionTypeResult.items():
        print(key)
            # print(key)
            # print(value)

    # Load Grade

    # print(train_split)
    # print(test_split)
    # path = 'test_keys.pkl'
    # f = open(path, 'rb')
    # data = pickle.load(f)
