import math
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error

def AreMeansSame(l1, l2):
    '''
    H0 : Means are same
    H1 : Means are not same
    '''
    stat, p = ttest_ind(l1, l2)
    if p<=0.05:
        # Reject H0
        return False
    else:
        return None



def PointBreak(df1, df2, nRollingWindowLength):
    # Prove that there is a srtuctural break
    lstFailurePoints = []
    lstSuccessPoints = []
    for inx in range(len(df2)-nRollingWindowLength):
        if AreMeansSame(df1['engineSize'], df2['engineSize'][inx:inx+nRollingWindowLength]) is False:
            lstFailurePoints.append(inx)
        else:
            lstSuccessPoints.append(inx)
    nBreakPointPosition = 0
    for inx in lstFailurePoints:
        if nBreakPointPosition == inx:
            nBreakPointPosition += 1
        else:
            break
    return lstFailurePoints[nBreakPointPosition]

finalDf = pd.read_csv("data/sortedfilledprices.csv")
midlength=15969 #midlength = math.floor(len(finalDf)/2)
modelDf = finalDf[:midlength]
inferenceData = finalDf[midlength:]


modelDf_X = modelDf[['year', 'mileage', 'tax', 'mpg', 'engineSize']].values
modelDf_Y = modelDf['price'].values
modelDf_Y = modelDf_Y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(modelDf_X, modelDf_Y)

model = DecisionTreeRegressor(max_depth=4)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae_model = mean_absolute_error(y_test, y_pred)
r2model = r2_score(y_test, y_pred)

print("Model r2:{} and MAE:{}".format(r2model, mae_model))


for nRollingWindowLength in (1000, 2000, 2500, 5000):
    nPointBreak = PointBreak(modelDf, inferenceData, nRollingWindowLength)
    rollingDf = inferenceData[:nPointBreak]
    rollingX = rollingDf[['year', 'mileage', 'tax', 'mpg', 'engineSize']].values
    rollingY = rollingDf['price'].values
    rollingY = rollingY.reshape(-1, 1)
    rolling_pred = model.predict(rollingX)
    mae_inference = mean_absolute_error(rollingY, rolling_pred)
    r2_inference = r2_score(rollingY, rolling_pred)
    print("Inference StructuralBreak:{} r2:{} and MAE:{}".format(nPointBreak, r2_inference, mae_inference))

# stat, p = ttest_ind(h1df['mileage'], h2df['mileage'])

# finalDf = pd.read_csv("data/final.csv")
# finalDf = finalDf[["year","mileage","tax","mpg","engineSize","price"]]
# finalDf = finalDf.dropna()
# finalDf = finalDf.sort_values(by=['year'])
# midlength = math.floor(len(finalDf)/2)
# h1df = finalDf[:midlength]
# h2df = finalDf[midlength:]


# if __name__=="__main__":
#     filenames = ["data/audi.csv",
#                 "data/cclass.csv",
#                 "data/ford.csv",
#                 "data/merc.csv",
#                 "data/toyota.csv",
#                 #"data/unclean focus.csv",
#                 "data/vw.csv",
#                 "data/bmw.csv",
#                 "data/focus.csv",
#                 "data/hyundi.csv",
#                 "data/skoda.csv",
#                 #"data/unclean cclass.csv",
#                 "data/vauxhall.csv"]


#     dfList = []

#     years = set()
#     for file in filenames:
#         df = pd.read_csv(file)
#         strMake = file[5:][:-4]
#         df['make'] = [strMake]*df.shape[0]
#         dfList.append(df)

#     finalDf = pd.concat(dfList)
#     finalDf.to_csv("data/final.csv")


