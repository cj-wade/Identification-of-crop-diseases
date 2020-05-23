import pickle
def saveStruct(model,path):
    json_string = model.to_json()
    open(path, 'w').write(json_string)
    
def saveHistory(history,path):
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)




