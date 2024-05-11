import kaggle_preprocessing

# 9. MODELLING 
# 9.1 Introduction 
# # VARIABLES INDEPENDIENTES
x = train.drop(['SalePrice'], axis = 1)

# # VARIABLE OBJETIVO
y = train['SalePrice']
# 9.2 Cross-Validation 
kf = KFold(n_splits=10 , shuffle=True, random_state=42)

def cv_rmse(model, x=x):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 9.3 Models 
# #####################################
# ### Ridge, Lasso, ElasticNet, SVR ###
#####################################

alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=kf))

alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000000, alphas=alphas_lasso, cv=kf))


alphas_elastic = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
l1ratio_elastic = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000000, alphas=alphas_elastic, l1_ratio=l1ratio_elastic, cv=kf))

svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))


# #################
# ### Light GBM ###
# #################

lightgbm = make_pipeline(RobustScaler(),LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=9000, max_bin=200, 
                                                      bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.2, 
                                                      feature_fraction_seed=7, verbose=-1))

# ################
# ### Catboost ###
################

catboost= make_pipeline(RobustScaler(), CatBoostRegressor(iterations=6000, learning_rate=0.005, depth=4, l2_leaf_reg=1, eval_metric='RMSE',
                                                          early_stopping_rounds=200, verbose=False))
# Model Scoring 
score_ridge = cv_rmse(ridge)
score_lasso = cv_rmse(lasso)
score_elastic = cv_rmse(elasticnet)
score_svr = cv_rmse(svr)
score_lightgbm = cv_rmse(lightgbm)
score_catboost = cv_rmse(catboost)

scores=[score_ridge.mean(), score_lasso.mean(), score_elastic.mean(), score_svr.mean(), score_lightgbm.mean(), score_catboost.mean()]
stds=[score_ridge.std(), score_lasso.std(), score_elastic.std(), score_svr.std(), score_lightgbm.std(), score_catboost.std()]
# Scoring Information 
models=['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'LightGBM', 'Catboost']

rmse_scores = {'RMSE': scores, 'Deviation': stds} 
rmse_scores_df = pd.DataFrame(rmse_scores, index=models)
rmse_scores_df
# RMSE	Desviación
# Ridge	0.125990	0.024634
# Lasso	0.123903	0.025329
# ElasticNet	0.123694	0.025191
# SVR	0.122883	0.027047
# LightGBM	0.121370	0.016789
# Catboost	0.117598	0.017633
# Stacking 
# ###############
# ### Stacked ###
###############

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, lightgbm), meta_regressor=lightgbm, use_features_in_secondary=True)
# Blending Models 
elastic_model_full_data = elasticnet.fit(x, y)
lasso_model_full_data = lasso.fit(x, y)
ridge_model_full_data = ridge.fit(x, y)
svr_model_full_data = svr.fit(x, y)
lgb_model_full_data = lightgbm.fit(x, y)
catboost_model_full_data = catboost.fit(x, y)
stack_model_full_data = stack_gen.fit(np.array(x), np.array(y))

def blend_models_predict(x):
    return ((0.16 * catboost_model_full_data.predict(x)) + \
            (0.16 * lasso_model_full_data.predict(x)) + \
            (0.11 * ridge_model_full_data.predict(x)) + \
            (0.1 * svr_model_full_data.predict(x)) + \
            (0.2 * lgb_model_full_data.predict(x)) + \
            (0.27 * stack_model_full_data.predict(np.array(x))))

print(rmsle(y, blend_models_predict(x)))
# MODEL DEVIATION:
# 0.07893344299841797
# CSV submission preparation 
submission=pd.read_csv('sample_submission.csv')
x_test=test.values
y_pred = (np.expm1(blend_models_predict(x_test)))
submission.iloc[:,1]=y_pred
submission.head(10)

y_pred = pd.DataFrame(y_pred, columns=['Prediction'])
y_pred.to_csv('result.csv', index=False) 
# Id	SalePrice
# 0	1461	120011.211386
# 1	1462	156836.688186
# 2	1463	183546.277644
# 3	1464	196981.912588
# 4	1465	184201.757098
# 5	1466	171844.862488
# 6	1467	178028.568747
# 7	1468	166192.723361
# 8	1469	183819.829525
# 9	1470	122350.625459
# submission.to_csv("submission_v6.csv", index=False)