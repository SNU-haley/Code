import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

case = '1_Basic_1_Seoul'

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

            ##하이퍼 파라미터 최적화

            from scipy.stats import randint, uniform
            from sklearn.pipeline import make_pipeline
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import RandomizedSearchCV

            pipe = make_pipeline(RandomForestRegressor())

            dists = {
                'randomforestregressor__n_estimators': randint(50, 500),
                'randomforestregressor__max_depth': [5, 10, 15, 20, None],
                'randomforestregressor__max_features': uniform(0, 1),
                'randomforestregressor__min_samples_leaf': [4, 8],
                'randomforestregressor__min_samples_split': [4, 8]}

            model = RandomizedSearchCV(
                pipe,
                param_distributions=dists,
                n_iter=50,
                cv=3,
                # scoring='accuracy',
                verbose=2,
                n_jobs=-1)

            model.fit(x_train, y_train)

            # 위에서 찾은 최적 하이퍼파라미터를 적용한 모델 생성.
            pipe = model.best_estimator_

            y_predicted = pipe.predict(x_test)
            evaluation = pipe.score(x_test, y_test)

            f = open(name + '.txt', 'w')
            f.write(f"""
                The hyperparameter search is complete. 
                The best hyperparameter: {pipe}.
                R2 = {evaluation}.
                """)
            f.close()

    # rescaling
    # x = x' * (max-min) + min
    # saving scaling factor in [max-min, min, max]

            y_predicted_total = model.predict(np.array(data_wodate_scaled.drop(columns=target)))
            y_predicted_total = pd.DataFrame(y_predicted_total, columns=target)

            for c in y_predicted_total:
                y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

                y_predicted_total.to_csv(name + '.csv', index=False)

