# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 20:55
# @Author  : 朝天椒
# @公众号  : 辣椒哈哈
# @FileName: gbm_xgb
# @Software: PyCharm

from base_parameter import *

class xgb_model:
    def __init__(self, train_data, val_data, label_name, val_need_columns, params, is_plot, is_kfold=False):
        '''
       the function of construct parameter introduction
       :param train_data: 训练数据, type：pd.DataFrame
       :param val_data: 验证数据, type: pd.DateFrame
       :param label_name: 标签名字, type: String
       :param val_need_columns: 需要返回的数据的列名, type: String
       :param params: 模型参数设置参数, type: dict
       :param is_plot: 是否在训练的时候保存训练与测试集的评价指标的图像变化情况, type: bool
       :param is_kfold: 是否对模型进行交叉验证的方式进行训练, type: bool
       '''
        self.train_data = train_data
        self.val_data = val_data
        self.val_need_columns = val_need_columns
        self.label_name = label_name

        self.params = params
        self.is_kfold = is_kfold
        self.is_plot = is_plot

        # 获取train_label
        self.train_label = self.train_data[self.label_name]
        self.train_data.drop(self.label_name, axis=1, inplace=True)
        print(f"the train_values shape is:{self.train_data.shape}")

        # 特征重要性variable
        self.feature_name = list(self.train_data.columns)
        self.feature_importance_values = np.zeros(len(self.feature_name))


    def get_best_boosttree(self):
        xgb_train = xgb.DMatrix(self.train_data, self.train_label)
        cv_results = xgb.cv(self.params, train_set=xgb_train, stratified=False, shuffle=True, metrics='l1',
                            show_stdv=True, seed=0, nfold=5,verbose_eval=20, early_stopping_rounds=150
                            )
        print(f"the best n_estimators is :{cv_results.shape[0]}")

    def model_raw(self):

        if self.is_kfold:
            test_accuracy_all = []
            val_predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            self.train_data = self.train_data.values
            self.train_label = self.train_label.values
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                print(f"is the {index} train!")
                train_x, test_x, train_y, test_y = self.train_data[train_index], self.train_data[test_index], \
                                                   self.train_label[train_index], self.train_label[test_index]
                xgb_train = xgb.DMatrix(train_x, train_y)
                xgb_test = xgb.DMatrix(test_x, test_y, reference=xgb_train)
                watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]

                xgb_model = xgb.train(self.params, dtrain=xgb_train, evals=watch_list, verbose_eval=20,
                                      early_stopping_rounds=150)

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                # test_predict = xgb_model.predict(test_x)
                # test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                # print(f"the k:{index} test_accuracy is :{test_accuracy}")
                # test_accuracy_all.append(test_accuracy)

                val_predict = xgb_model.predict(self.val_data.values)
                val_predict_all.append(val_predict)
                self.feature_importance_values += xgb_model.feature_importance()

            # 验证集预测值保存
            val_predict_all = np.array(val_predict_all)
            val_predict_all = np.mean(val_predict_all, axis=0)
            self.val_data[self.label_name] = list(val_predict_all.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./xgb_raw_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_importances = pd.DataFrame(
                {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            print(f"the feature importances is:{sorted(feature_importances.items(), key=lambda x: x[1])}")
            feature_importances.to_csv('./xgb_raw_cv_feature_inportances.csv', index=False)

            # 使用自定义的评价函数来评价模型的拟合效果
            # test_accuracy_all = np.array(test_accuracy_all)
            # test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            # test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            # print(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            # 单模型不使用交叉验证，直接使用简单的一次train，test
            random_value = random.randint(0, 10000)
            print(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label,
                                                                test_size=0.2, random_state=random_value)
            xgb_train = xgb.DMatrix(train_x, train_y)
            xgb_test = xgb.DMatrix(test_x, test_y, reference=xgb_train)
            watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]

            xgb_model = xgb.train(self.params, dtrain=xgb_train, evals=watch_list, verbose_eval=20,
                                  early_stopping_rounds=150)

            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            # test_predict = gbm_model.predict(test_x)
            # test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            # print(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            val_predict = xgb_model.predict(self.val_data.values)
            self.val_data[self.label_name] = list(val_predict.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./xgb_raw_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = gbm_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            print(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv('./xgb_raw_feature_inportances.csv', index=False)


    def model_sklearn(self):
        if self.is_kfold:
            test_accuracy_all = []
            val_predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                print(f"is the {index} train!")
                xgb_model = xgb.XGBRegressor(**self.params)
                xgb_model.fit(x=self.train_data[train_index], y=self.train_label[train_index], verbose=20,
                                        eval_metric="mae",
                                        eval_set=[(self.train_data[train_index], self.train_label[train_index]),
                                                  (self.train_data[test_index], self.train_label[test_index])],
                                        early_stopping_rounds=200)

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                # test_predict =  xgb_model.predict(self.train_data[test_index], num_iteration= xgb_model.best_iteration_)
                # test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                # print(f"the k:{index} test_accuracy is :{test_accuracy}!")
                # test_accuracy_all.append(test_accuracy)

                val_predict = xgb_model.predict(self.val_data.values)
                val_predict_all.append(val_predict)

            # 验证集预测值保存
            val_predict_all = np.array(val_predict_all)
            val_predict_all = np.mean(val_predict_all, axis=0)
            self.val_data[self.label_name] = list(val_predict_all.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./xgb_sklearn_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            self.feature_importance_values += xgb_model.feature_importances_
            feature_importances = pd.DataFrame(
                {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            feature_importances.to_csv('./xgb_sklearn_feature_inportances.csv', index=False)

            # 使用自定义的评价函数来评价模型的拟合效果
            # test_accuracy_all = np.array(test_accuracy_all)
            # test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            # test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            # print(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            random_value = random.randint(0, 10000)
            print(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label, test_size=0.2, random_state=random_value)
            xgb_model = xgb.XGBRegressor(**self.params)
            xgb_model.fit(train_x, train_y,
                          verbose=20, eval_metric="l1",
                          eval_set=[(train_x, train_y), (test_x, test_y)],
                          early_stopping_rounds=150
                        )
            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            # test_predict = xgb_model.predict(test_x)
            # test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            # print(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            val_predict = xgb_model.predict(self.val_data.values)
            self.val_data[self.label_name] = list(val_predict.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./xgb_sklearn_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = xgb_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            print(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv('./xgb_sklearn_feature_inportances.csv', index=False)



class gbm_model:
    def __init__(self, train_data, val_data, label_name, val_need_columns,  params, is_plot, is_kfold=False):
        '''
        the function of construct parameter introduction
        :param train_data: 训练数据, type：pd.DataFrame
        :param val_data: 验证数据, type: pd.DateFrame
        :param label_name: 标签名字, type: String
        :param val_need_columns: 需要返回的数据的列名, type: String
        :param params: 模型参数设置参数, type: dict
        :param is_plot: 是否在训练的时候保存训练与测试集的评价指标的图像变化情况, type: bool
        :param is_kfold: 是否对模型进行交叉验证的方式进行训练, type: bool
        '''
        self.train_data = train_data
        self.val_data = val_data
        self.val_need_columns = val_need_columns
        self.label_name = label_name

        self.params = params
        self.is_kfold = is_kfold
        self.is_plot = is_plot

        # 获取train_label
        self.train_label = self.train_data[self.label_name]
        self.train_data.drop(self.label_name, axis=1, inplace=True)
        print(f"the train_values shape is:{self.train_data.shape}")

        # 特征重要性variable
        self.feature_name = list(self.train_data.columns)
        self.feature_importance_values = np.zeros(len(self.feature_name))

    def get_best_boosttree(self):
        # cv选择模型的最佳的n_estimators个数
        lgb_train = lgb.Dataset(self.train_data.values, self.train_label.values)
        cv_results = lgb.cv(self.params, train_set=lgb_train, stratified=False, shuffle=True, metrics='l1',
                            show_stdv=True, seed=0, nfold=5,
                            verbose_eval=20, early_stopping_rounds=150)
        print(f"the cv model parameter is:{cv_results}")
        print('best n_estimators:', len(cv_results['l1-mean']))
        print(f"and the cv_results key is:{cv_results.keys()}, and best cv score:{cv_results['l1-mean'][-1]}")

    def model_rawapi(self):

        if self.is_kfold:
            test_accuracy_all = []
            val_predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            self.train_data = self.train_data.values
            self.train_label = self.train_label.values
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                print(f"is the {index} train!")
                train_x, test_x, train_y, test_y = self.train_data[train_index], self.train_data[test_index], self.train_label[train_index], self.train_label[test_index]
                lgb_train = lgb.Dataset(train_x, train_y)
                lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)

                gbm_model = lgb.train(self.params, train_set=lgb_train, valid_sets=[lgb_train, lgb_test],
                                      verbose_eval=20, early_stopping_rounds=150)
                
                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                # test_predict = gbm_model.predict(test_x)
                # test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                # print(f"the k:{index} test_accuracy is :{test_accuracy}")
                # test_accuracy_all.append(test_accuracy)
                
                val_predict = gbm_model.predict(self.val_data.values)
                val_predict_all.append(val_predict)
                self.feature_importance_values += gbm_model.feature_importance()

            # 验证集预测值保存
            val_predict_all = np.array(val_predict_all)
            val_predict_all = np.mean(val_predict_all, axis=0)
            self.val_data[self.label_name] = list(val_predict_all.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./raw_cv_val_predict.csv', index=False, encoding='utf-8')
            
            # 特征重要性的保存
            feature_importances = pd.DataFrame(
                                    {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            print(f"the feature importances is:{sorted(feature_importances.items(), key=lambda x: x[1])}")
            feature_importances.to_csv('./gbm_raw_cv_feature_inportances.csv', index=False)
            
            # 使用自定义的评价函数来评价模型的拟合效果
            # test_accuracy_all = np.array(test_accuracy_all)
            # test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            # test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            # print(f"the test_accuracy_avg is:{test_accuracy_avg}")
           
        else:

            # 单模型不使用交叉验证，直接使用简单的一次train，test
            random_value = random.randint(0, 10000)
            print(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label, 
                                                                test_size=0.2, random_state=random_value)
            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)

            gbm_model = lgb.train(self.params, train_set=lgb_train, valid_sets=[lgb_train, lgb_test],
                                    verbose_eval=20, early_stopping_rounds=150)

            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            # test_predict = gbm_model.predict(test_x)
            # test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            # print(f"the test_accuracy is :{test_accuracy}!")
            
            # 验证集预测值保存
            val_predict = gbm_model.predict(self.val_data.values)
            self.val_data[self.label_name] = list(val_predict.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./raw_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = gbm_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            print(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x:x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv('./gbm_raw_feature_inportances.csv', index=False)

    def model_sklearn(self):
        if self.is_kfold:
            test_accuracy_all = []
            val_predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                print(f"is the {index} train!")
                gbm_model = lgb.LGBMRegressor(**self.params)
                gbm_model.fit(self.train_data[train_index], self.train_label[train_index], verbose=10,
                                        eval_metric="mae",
                                        eval_set=[(self.train_data[train_index], self.train_label[train_index]),
                                                    (self.train_data[test_index], self.train_label[test_index])],
                                        early_stopping_rounds=200)

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                # test_predict =  gbm_model.predict(self.train_data[test_index], num_iteration= gbm_model.best_iteration_)
                # test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                # print(f"the k:{index} test_accuracy is :{test_accuracy}!")
                # test_accuracy_all.append(test_accuracy)

                val_predict =  gbm_model.predict(self.val_data.values)
                val_predict_all.append(val_predict)

            # 验证集预测值保存
            val_predict_all = np.array(val_predict_all)
            val_predict_all = np.mean(val_predict_all, axis=0)
            self.val_data[self.label_name] = list(val_predict_all.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./gbm_sklearn_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            self.feature_importance_values += gbm_model.feature_importances_
            feature_importances = pd.DataFrame(
                {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            feature_importances.to_csv('./gbm_sklearn_feature_inportances.csv', index=False)

            # 使用自定义的评价函数来评价模型的拟合效果
            # test_accuracy_all = np.array(test_accuracy_all)
            # test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            # test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            # print(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            random_value = random.randint(0, 10000)
            print(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label, test_size=0.2, random_state=random_value)
            gbm_model = lgb.LGBMRegressor(**self.params)
            gbm_model.fit(train_x, train_y,
                          verbose=20, eval_metric="l1",
                          eval_set=[(train_x, train_y), (test_x, test_y)],
                          early_stopping_rounds=150
                        )
            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            # test_predict = gbm_model.predict(test_x)
            # test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            # print(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            val_predict = gbm_model.predict(self.val_data.values)
            self.val_data[self.label_name] = list(val_predict.reshape(-1))
            self.val_data = self.val_data[[self.val_need_columns, self.label_name]]
            self.val_data.to_csv('./raw_cv_val_predict.csv', index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = gbm_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            print(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv('./gbm_raw_feature_inportances.csv', index=False)

if __name__ == '__main__':
    print(f"the start :!")
    train_datas, val_datas = data_clean(train_path, val_path)
    # model_sklearn(train_datas, val_datas, is_kfold=False)
    model_gbm = gbm_model(train_datas, val_datas, val_need_columns='uid',
                          label_name='label', is_kfold=False, params=gbm_params)
    # model_gbm.get_best_boosttree()
    model_gbm.model_sklearn()





