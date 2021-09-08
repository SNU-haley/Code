import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


case = '1_Basic_2_BR'

df = pd.read_csv('C:\\Users\\Haley\\Dropbox\\패밀리룸\\MVI\\Data\\'+case+'_raw.csv')
scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max()-df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min())/denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

# seeds = [777, 1004, 322, 224, 417]
seeds = [777, 1004, 322]
ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals,ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals','ions-ocec-elementals']

iteration = 1

for s in range(len(seeds)):
    for ele in range(len(elements)):
        for iter in range(iteration):

            name = case + '_result_'+ str(seeds[s])+'_RF2_'+str(elements_name[ele])+'_'+str(iter+1)

            eraser = df.sample(int(len(df)*0.2), random_state=seeds[s]).index
            target = elements[ele]

            x_train = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=target))
            y_train = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, target]
            y_train = np.array(y_train)

            val_splitrate = 0.2
            vals = np.random.choice(x_train.shape[0], int(x_train.shape[0] * val_splitrate), replace=False)
            vals.sort()

            x_val = x_train[vals]
            y_val = y_train[vals]

            x_train = np.delete(x_train, [vals], axis=0)
            y_train = np.delete(y_train, [vals], axis=0)

            x_test = np.array(data_wodate_scaled.loc[eraser].drop(columns=target))
            y_test = np.array(data_wodate_scaled.loc[eraser, target])

            #Random_forrest_regressor
            # max_depth = 4
            # min_samples_leaf = 18
            # min_samples_split = 8
            # n_estimators = 20
            # model = RandomForestRegressor(n_estimators=n_estimators,
            #                               max_depth=max_depth,
            #                               min_samples_leaf=min_samples_leaf,
            #                               min_samples_split=min_samples_split)
            model = RandomForestRegressor()
            model.fit(x_train, y_train)


            #결과예측
            y_pred = model.predict(x_test)
            from sklearn.metrics import r2_score
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import mean_absolute_error
            # 결과살펴보자
            # print('###RF>N_estimators:',n_estimators)
            print('MAE:',mean_absolute_error(y_test,y_pred))
            print('MSE:',mean_squared_error(y_test, y_pred))
            print('R2:', format(r2_score(y_test,y_pred)))

            y_predicted_total = model.predict(np.array(data_wodate_scaled.drop(columns=target)))
            y_predicted_total = pd.DataFrame(y_predicted_total, columns=target)

            for c in y_predicted_total:
                y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

                y_predicted_total.to_csv(name+'.csv', index=False)

