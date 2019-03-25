class xgb_model:
    def __init__(self, train_data, predict_data, param,
                 label_name='helo',
                 predict_name='hello_new',
                 model_name='model_sklearn',
                 is_kfold=False):
        logger.debug("model parameter init:")
        self.model_name = model_name
        self.train_data = train_data
        self.predict_data = predict_data
        self.predict_result = self.predict_data.copy()
        self.params = param
        self.label_name = label_name
        self.predict_name = predict_name
        self.is_kfold = is_kfold
        logger.debug("init is over, the model is run:")
        self.run()

    def data_clean(self):
        # 获取train_label
        self.train_label = self.train_data[self.label_name].values
        self.train_data.drop(self.label_name, axis=1, inplace=True)
        logger(f"the train data shape is:{self.train_data.shape}, "
               f"and the predict data shape is:{self.predict_data.shape}")

        # 特征重要性variable
        self.feature_name = list(self.train_data.columns)
        self.feature_importance_values = np.zeros(len(self.feature_name))

        self.train_data = self.train_data.values
        self.predict_data = self.predict_data.values
        if self.is_log:
            self.train_label = np.log(self.train_label)

    def run(self):
        self.data_clean()
        if self.model_name == 'model_sklearn':
            self.model_sklearn()
        else:
            self.model_rawapi()

    def get_best_boosttree(self):
        xgb_train = xgb.DMatrix(self.train_data, self.train_label)
        cv_results = xgb.cv(self.params, train_set=xgb_train, stratified=False, shuffle=True, metrics='l1',
                            show_stdv=True, seed=0, nfold=5, verbose_eval=20, early_stopping_rounds=150
                            )
        print(f"the best n_estimators is :{cv_results.shape[0]}")

    def model_raw(self):
        if self.is_kfold:
            test_accuracy_all = []
            predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                logger.debug(f"is the {index} train!")
                train_x, test_x, train_y, test_y = self.train_data[train_index], self.train_data[test_index], \
                                                   self.train_label[train_index], self.train_label[test_index]
                xgb_train = xgb.DMatrix(train_x, train_y)
                xgb_test = xgb.DMatrix(test_x, test_y, reference=xgb_train)
                watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]

                xgb_model = xgb.train(self.params, dtrain=xgb_train, evals=watch_list, verbose_eval=20,
                                      early_stopping_rounds=150)
                joblib.dump(lgb, os.path.join(param_help.model_data_dir, f'xgb_raw_model_cv_{index}.pkl'))

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                test_predict = xgb_model.predict(test_x)
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                logger.debug(f"the k:{index} test_accuracy is :{test_accuracy}")
                test_accuracy_all.append(test_accuracy)

                predict_y = xgb_model.predict(self.predict_data)
                predict_all.append(predict_y)
                self.feature_importance_values += xgb_model.feature_importance()

            # 预测值保存
            predict_all = np.array(predict_all)
            predict_all = np.mean(predict_all, axis=0)
            self.predict_result[self.label_name] = list(predict_all.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'xgb_raw_cv_val_predict.csv'),
                                       index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_importances = pd.DataFrame(
                {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            logger.debug(f"the feature importances is:{sorted(feature_importances.items(), key=lambda x: x[1])}")
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir, 'xgb_raw_cv_feature_importances.csv',
                                                    index=False, encoding='utf-8'))

            # 使用自定义的评价函数来评价模型的拟合效果
            test_accuracy_all = np.array(test_accuracy_all)
            test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            logger.debug(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            # 单模型不使用交叉验证，直接使用简单的一次train，test
            random_value = random.randint(0, 10000)
            logger.debug(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label,
                                                                test_size=0.2, random_state=random_value)
            logger.debug(f"the test_x:{np.shape(test_x)}, and the test_y is:{np.shape(test_y)}")

            xgb_train = xgb.DMatrix(train_x, train_y)
            xgb_test = xgb.DMatrix(test_x, test_y, reference=xgb_train)
            watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]

            xgb_model = xgb.train(self.params, dtrain=xgb_train, evals=watch_list, verbose_eval=20,
                                  early_stopping_rounds=150)
            joblib.dump(xgb_model, os.path.join(param_help.model_data_dir, 'xgb_raw_simple_model.pkl'))

            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            test_predict = xgb_model.predict(test_x)
            test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            logger.debug(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            predict_y = xgb_model.predict(self.predict_data)
            self.predict_result[self.predict_name] = list(predict_y.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir,'xgb_raw_cv_val_predict.csv',
                                                    index=False, encoding='utf-8'))

            # 特征重要性的保存
            feature_values = xgb_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            logger.debug(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir,'xgb_raw_feature_importances.csv', index=False))

    def model_sklearn(self):
        if self.is_kfold:
            test_accuracy_all = []
            predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                logger.debug(f"is the {index} train!")
                xgb_model = xgb.XGBRegressor(**self.params)
                xgb_model.fit(x=self.train_data[train_index], y=self.train_label[train_index], verbose=20,
                              eval_metric="mae",
                              eval_set=[(self.train_data[train_index], self.train_label[train_index]),
                                        (self.train_data[test_index], self.train_label[test_index])],
                              early_stopping_rounds=200)

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                test_predict =  xgb_model.predict(self.train_data[test_index], num_iteration= xgb_model.best_iteration_)
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                logger.debug(f"the k:{index} test_accuracy is :{test_accuracy}!")
                test_accuracy_all.append(test_accuracy)

                predict_y = xgb_model.predict(self.predict_data)
                predict_all.append(predict_y)

            # 验证集预测值保存
            predict_all = np.array(predict_all)
            predict_all = np.mean(predict_all, axis=0)
            self.predict_result[self.predict_name] = list(predict_all.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'xgb_sklearn_cv_val_predict.csv'),
                                 index=False, encoding='utf-8')

            # 特征重要性的保存
            self.feature_importance_values += xgb_model.feature_importances_
            feature_importances = pd.DataFrame(
                {'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir, 'xgb_sklearn_feature_inportances.csv'),
                                                    index=False, encoding='utf-8')

            # 使用自定义的评价函数来评价模型的拟合效果
            test_accuracy_all = np.array(test_accuracy_all)
            test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            logger.debug(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            random_value = random.randint(0, 10000)
            logger.debug(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label, test_size=0.2,
                                                                random_state=random_value)
            xgb_model = xgb.XGBRegressor(**self.params)
            xgb_model.fit(train_x, train_y,
                          verbose=20, eval_metric="l1",
                          eval_set=[(train_x, train_y), (test_x, test_y)],
                          early_stopping_rounds=150
                          )
            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            test_predict = xgb_model.predict(test_x)
            test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            logger.debug(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            predict_y = xgb_model.predict(self.predict_data)
            self.predict_result[self.predict_name] = list(predict_y.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir,'xgb_sklearn_cv_val_predict.csv'),
                                              index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = xgb_model.feature_importance()
            feature_dicts = dict(zip(self.feature_name, feature_values))
            logger.debug(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir,'xgb_sklearn_feature_inportances.csv'),
                                                    index=False, encoding='utf-8')

class gbm_model:
    def __init__(self, train_data, predict_data, param,
                 label_name='hello',
                 predict_name='hello_new',
                 model_name='model_sklearn',
                 is_kfold=False):
        logger.debug("model parameter init:")
        self.model_name = model_name
        self.train_data = train_data
        self.predict_data = predict_data
        self.predict_result = self.predict_data.copy()
        self.params = param
        self.label_name = label_name
        self.predict_name = predict_name
        self.is_kfold = is_kfold
        logger.debug("init is over, the model is run:")
        self.run()

    def data_clean(self):
        # 获取train_label
        self.train_label = self.train_data[self.label_name].values
        self.train_data.drop(self.label_name, axis=1, inplace=True)
        logger(f"the train data shape is:{self.train_data.shape}, "
               f"and the predict data shape is:{self.predict_data.shape}")

        # 特征重要性variable
        self.feature_name = list(self.train_data.columns)
        self.feature_importance_values = np.zeros(len(self.feature_name))

        self.train_data = self.train_data.values
        self.predict_data = self.predict_data.values
        if self.is_log:
            self.train_label = np.log(self.train_label)

    def run(self):
        self.data_clean()
        if self.model_name == 'model_sklearn':
            self.model_sklearn()
        else:
            self.model_rawapi()

    def get_best_boosttree(self):
        # cv选择模型的最佳的n_estimators个数
        lgb_train = lgb.Dataset(self.train_data, self.train_label)
        cv_results = lgb.cv(self.params, train_set=lgb_train, stratified=False, shuffle=True, metrics='l1',
                            show_stdv=True, seed=0, nfold=5,
                            verbose_eval=20, early_stopping_rounds=gbm_params['num_iterations']/10)
        logger.debug(f"the cv model parameter is:{cv_results}")
        logger.debug('best n_estimators:', len(cv_results['l1-mean']))
        logger.debug(f"and the cv_results key is:{cv_results.keys()}, and best cv score:{cv_results['l1-mean'][-1]}")

    def model_rawapi(self):

        if self.is_kfold:
            test_accuracy_all = []
            predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                logger.debug(f"is the {index} train!")
                train_x, test_x, train_y, test_y = self.train_data[train_index], self.train_data[test_index], \
                                                   self.train_label[train_index], self.train_label[test_index]
                lgb_train = lgb.Dataset(train_x, train_y)
                lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)

                gbm_model = lgb.train(self.params, train_set=lgb_train, valid_sets=[lgb_train, lgb_test],
                                      verbose_eval=20, early_stopping_rounds=200)
                joblib.dump(lgb, os.path.join(param_help.model_data_dir, f'gbm_raw_model_cv_{index}.pkl'))

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
                test_predict = gbm_model.predict(test_x)
                if self.is_log:
                    test_predict = np.exp(test_predict)
                    test_accuracy = get_accuracy(y_predict=test_predict, y_real=np.expm1(self.train_label[test_index]))
                else:
                    test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                logger.debug(f"the k:{index} test_accuracy is :{test_accuracy}!")
                test_accuracy_all.append(test_predict)

                predict_y = gbm_model.predict(self.predict_data)
                if self.is_log:
                    predict_y = np.exp(self.predict_y)
                predict_all.append(predict_y)
                self.feature_importance_values += gbm_model.feature_importance()

            # 预测值保存
            predict_all = np.array(predict_all)
            predict_all = np.mean(predict_all, axis=0)
            self.predict_result[self.label_name] = list(predict_all.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_raw_cv_dispatch_time_predict.csv'),
                                       index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            logger.debug(f"the feature importances is:{sorted(feature_importances.items(), key=lambda x: x[1])}")
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_feature_inportances.csv'),
                                       index=False, encoding='utf-8')

            # 使用自定义的评价函数来评价模型的拟合效果
            test_accuracy_all = np.array(test_accuracy_all)
            test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            logger.debug(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            # 单模型不使用交叉验证，直接使用简单的一次train，test
            random_value = random.randint(0, 10000)
            logger.debug(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label,
                                                                test_size=0.2, random_state=random_value)
            logger.debug(f"the test_x:{np.shape(test_x)}, and the test_y is:{np.shape(test_y)}")

            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)

            gbm_model = lgb.train(self.params, train_set=lgb_train, valid_sets=[lgb_train, lgb_test],
                                  verbose_eval=20, early_stopping_rounds=150)
            joblib.dump(lgb, os.path.join(param_help.model_data_dir, 'gbm_raw_simple_model.pkl'))
            logger.debug(f"the model parameter is:{gbm_model.model_to_string()['parameters']}")

            # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数并启动注释代码
            test_predict = gbm_model.predict(test_x)
            if self.is_log:
                test_predict = np.exp(test_predict)
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=np.expm1(test_y))
            else:
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)
            logger.debug(f"the test_accuracy is :{test_accuracy}!")

            # 验证集预测值保存
            predict_y = gbm_model.predict(self.predict_data)
            if self.is_log:
                predict_y = np.exp(predict_y)
            self.predict_result[self.predict_name] = list(predict_y.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'raw_dispatch_time_predict.csv'),
                                         index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_values = gbm_model.feature_importance()
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': feature_values})
            feature_dicts = dict(zip(self.feature_name, feature_values))
            logger.debug(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x:x[1])}")
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_raw_feature_inportances.csv'),
                                       index=False)

    def model_sklearn(self):
        if self.is_kfold:
            test_accuracy_all = []
            predict_all = []
            kfold = KFold(n_splits=5, random_state=2018, shuffle=True)
            for index, (train_index, test_index) in enumerate(kfold.split(self.train_data, self.train_label)):
                logger.debug(f"is the {index} train!")
                lgb_model = lgb.LGBMRegressor(**self.params)
                lgb_model.fit(self.train_data[train_index], self.train_label[train_index], verbose=10,
                              eval_metric="mae",
                              eval_set=[(self.train_data[train_index], self.train_label[train_index]),
                                        (self.train_data[test_index], self.train_label[test_index])],
                                        early_stopping_rounds=150)
                joblib.dump(lgb_model, os.path.join(param_help.model_data_dir, f'gbm_sk_model_cv_{index}.pkl'))

                # 如果要自己定义相关的评价函数，就自己定义get_accuracy()函数
                test_predict = lgb_model.predict(self.train_data[test_index],
                                                 num_iteration=lgb_model.best_iteration_)
                predict_y = lgb_model.predict(self.predict_data)
                if self.is_log:
                    test_predict = np.exp(test_predict)
                    predict_y = np.exp(predict_y)
                    test_accuracy = get_accuracy(y_predict=test_predict, y_real=np.expm1(self.train_label[test_index]))
                else:
                    test_accuracy = get_accuracy(y_predict=test_predict, y_real=self.train_label[test_index])
                logger.debug(f"the k:{index} test_accuracy is :{test_accuracy}!")
                test_accuracy_all.append(test_predict)
                predict_all.append(predict_y)
                self.feature_importance_values += lgb_model.feature_importances_

            # 预测值保存
            predict_all = np.array(predict_all)
            predict_all = np.mean(predict_all, axis=0)
            self.predict_result[self.label_name] = list(predict_all.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_raw_cv_dispatch_time_predict.csv'),
                                        index=False, encoding='utf-8')

            # 特征重要性的保存
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            logger.debug(f"the feature importances is:{sorted(feature_importances.items(), key=lambda x: x[1])}")
            feature_importances.to_csv(os.path.join(param_help.model_data_dir, 'gbm_sklearn_feature_inportances.csv'),
                                       index=False, encoding='utf-8')

            # 使用自定义的评价函数来评价模型的拟合效果
            test_accuracy_all = np.array(test_accuracy_all)
            test_accuracy_all = np.mean(test_accuracy_all, axis=0)
            test_accuracy_avg = get_accuracy(y_predict=test_accuracy_all, y_real=self.train_label[test_index])
            logger.debug(f"the test_accuracy_avg is:{test_accuracy_avg}")
        else:
            random_value = random.randint(0, 10000)
            logger.debug(f"the random_value is:{random_value}")
            train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.train_label, test_size=0.2, random_state=random_value)
            logger.debug(f"the test_x:{np.shape(test_x)}, and the test_y is:{np.shape(test_y)}")
            lgb_model = lgb.LGBMRegressor(**self.params)

            lgb_model.fit(train_x, train_y,
                              verbose=20,
                              eval_metric = "l1",
                              eval_set = [(train_x, train_y), (test_x, test_y)],
                              early_stopping_rounds=200
                          )
            lgb_model.print_evaluation(period=10, show_stdv=True)
            joblib.dump(lgb_model, os.path.join(param_help.model_data_dir, 'gbm_sk_simple_model.pkl'))

            test_predict = lgb_model.predict(test_x)
            predict_y = lgb_model.predict(self.predict_data)

            if self.is_log:
                test_predict = np.exp(test_predict)
                predict_y = np.exp(predict_y)
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=np.expm1(test_y))
            else:
                test_accuracy = get_accuracy(y_predict=test_predict, y_real=test_y)

            logger.debug(f"the model parameter is:{lgb_model.get_params()}")
            logger.debug(f"the test_accuracy is :{test_accuracy}!")

            # 预测值保存
            self.predict_result[self.predict_name] = list(predict_y.reshape(-1))
            self.predict_result.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_sk_dispatch_time_predict.csv'),
                                         index=False, encoding='utf-8')

            # 特征重要性的保存
            self.feature_importance_values = lgb_model.feature_importances_
            feature_dicts = dict(zip(self.feature_name,  self.feature_importance_values))
            logger.debug(f"the feature importances is:{sorted(feature_dicts.items(), key=lambda x: x[1])}")
            feature_importances = pd.DataFrame({'feature': self.feature_name, 'importance': self.feature_importance_values})
            feature_importances.to_csv(os.path.join(param_help.predict_data_dir, 'gbm_sklearn_feature_inportances.csv'), index=False)
