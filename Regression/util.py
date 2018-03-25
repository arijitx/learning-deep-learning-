import pandas as pd
def generate_kaggle_submission(test,predictions,id_label,target_label):
    result=pd.DataFrame(test[id_label])
    result[target_label]=pd.Series(predictions.flatten())
    result.to_csv('submission.csv',index=False)
