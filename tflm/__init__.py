
def modelling(first, second, target, inter_portion=0.5, sqr_portion=0.5, contribution_portion=0.9, final_contribution=0.9, node_size=1.0, final=False, modeller="LightGBM"):
      
  cols_drop = [target ]

  def shap_frame(first,second, target, node_size):

    #could also use linear booster XGBoost
    d_train = lgb.Dataset(first.drop(columns=[target]), label=first[target])
    d_valid = lgb.Dataset(second.drop(columns=[target]), label=second[target])
    params = {
      
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmsle',
        'max_depth': 6, 
        'learning_rate': 0.1,
        'verbose': 0,
      'num_threads':16}
    n_estimators = 100

    model = lgb.train(params, d_train, 100, verbose_eval=1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(first.drop([target], axis=1))
    shap_fram = pd.DataFrame(shap_values[:,:], columns=list(first.drop([target], axis=1).columns))
    shap_new = shap_fram.sum().sort_values().to_frame()
    print("Finished TreeExplainer")
    
    return shap_new, explainer, model

  def shap_frame_keras(first,second, target, node_size):
    ## deepexplainer and gradientexplainer has no interaction_values

    # Set the input shape
    input_shape = (len(first.columns)-1,)
    print(f'Feature shape: {input_shape}')

    # Create the model
    model = Sequential()
    l_1 = int((first.shape[1]*2)/contribution_portion)
    l_2 = int((first.shape[1]*1)/contribution_portion)
    print(l_1)
    print(l_2)
    model.add(Dense(l_1, input_shape=input_shape, activation='relu'))
    model.add(Dense(l_2, activation='relu'))
    model.add(Dense(1, activation='linear'))


    # Configure the model and start training
    # validation_data=(X_test, Y_test)


    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.01), metrics=['mean_squared_error'])

    stoppy = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(first.drop(columns=[target]), first[target], epochs=180, batch_size=10, verbose=1, validation_split=0.2, callbacks=[stoppy])

    #explainer = shap.DeepExplainer(model,first.drop(columns=[target]).values)

    explainer  = shap.GradientExplainer(model,first.drop(columns=[target]).values)

    shap_values = explainer.shap_values(first.drop([target], axis=1).values)
    shap_fram = pd.DataFrame(shap_values[0][:,:], columns=list(first.drop([target], axis=1).columns))
    shap_new = shap_fram.sum().sort_values().to_frame()

    return shap_new, explainer, model

  if modeller=="Keras":
    shape_frame_all = shap_frame_keras
  else:
    shape_frame_all = shap_frame

  shap_new, explainer, model = shape_frame_all(first,second,target,node_size)

  shap_new_abs = shap_new[0].abs()
  shap_new_abs = shap_new_abs.sort_values(ascending=False)
  main_ft = shap_new_abs[shap_new_abs.cumsum().sub((shap_new_abs.sum()*sqr_portion)).le(0)]

  preds = model.predict(second.drop(columns=[target]))
  mse = mean_squared_error(second[target], preds)
  print(mse)

  def main_calc(new_df,main_ft):
    df_square = new_df[list(main_ft.index)]
    sqr_name = [str(fa)+"_POWER_2" for fa in df_square.columns]
    log_p_name = [str(fa)+"_LOG_p_one_abs" for fa in df_square.columns]
    rec_p_name = [str(fa)+"_RECIP_p_one" for fa in df_square.columns]
    sqrt_name = [str(fa)+"_SQRT_p_one" for fa in df_square.columns]

    df_sqr = pd.DataFrame(np.power(df_square.values, 2),columns=sqr_name, index=new_df.index)
    df_log = pd.DataFrame(np.log(df_square.add(1).abs().values),columns=log_p_name, index=new_df.index)
    df_rec = pd.DataFrame(np.reciprocal(df_square.add(1).values),columns=rec_p_name, index=new_df.index)
    df_sqrt = pd.DataFrame(np.sqrt(df_square.abs().add(1).values),columns=sqrt_name, index=new_df.index)

    dfs = [df_sqr, df_log, df_rec, df_sqrt]

    df_connect=  pd.concat(dfs, axis=1)

    return df_connect

  ## An attempt to decrease the amount of interactin features.

  select_ft = shap_new_abs[shap_new_abs.cumsum().sub((shap_new_abs.sum()*contribution_portion)).le(0)]
  select_ft = list(select_ft.index)
  select_ft.append(target)

  ## has to remain tree for interaction effects. shap_frame
  shap_select, explainer, model = shap_frame(first[select_ft], second[select_ft], target, node_size )
  shap_select_abs = shap_select[0].abs()

  ### Interactions Features
  shap_interaction_values = explainer.shap_interaction_values(first[select_ft].drop(cols_drop, axis=1))

  shap_interaction_values_abs = abs(shap_interaction_values)
  df_start = pd.DataFrame(np.sum(shap_interaction_values_abs ,axis=0),columns=first[select_ft].drop(cols_drop, axis=1).columns, index=first[select_ft].drop(cols_drop, axis=1).columns)

  #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
  sol = (df_start.where(np.triu(np.ones(df_start.shape), k=1).astype(np.bool))
                  .stack()
                  .sort_values(ascending=False))
  #first element of sol series is the pair with the bigest correlation

  ## New Data Frames From Feature Interaction and Main Feature
  ## If you have features [a, b, c] the default polynomial features(in sklearn the degree is 2) should be [1, a, b, c, a^2, b^2, c^2, ab, bc, ca].

  ## Interaction Calculations

  dab = sol[sol.cumsum().sub((sol.sum()*inter_portion)).le(0)]

  list_one = [da[0] for da in dab.index]
  list_two = [da[1] for da in dab.index]

  def inter_cal(list_one, list_two,new_df):

    mult = [str(ra)+"_X_"+str(ba) for ra, ba in zip(list_one, list_two)]
    div = [str(ra)+"_DIV_"+str(ba) for ra, ba in zip(list_one, list_two)]
    print("len one " + str(len(list_one)) )
    print("len two " + str(len(list_two)) )
    inter_mult = pd.DataFrame(new_df[list_one].values*new_df[list_two].values, columns=mult, index=new_df.index)
    div_p_one = pd.DataFrame(new_df[list_one].add(1).values/new_df[list_two].add(1).values, columns=div, index=new_df.index)

    df_one = pd.concat((inter_mult,div_p_one), axis=1)

    return df_one


  def combine(target, list_one, list_two, main_ft, new_df):

    inter_mult = inter_cal(list_one, list_two, new_df.drop(columns=[target]))

    df_sqr = main_calc(new_df.drop(columns=[target]),main_ft)

    new = pd.concat((inter_mult,df_sqr),axis=1)
    new2 = pd.concat((new_df,new),axis=1)

    new2 = new2.loc[:,~new2.columns.duplicated()]

    return new2

  new_first = combine(target, list_one, list_two, main_ft, first)

  new_second = combine(target, list_one, list_two, main_ft, second)

  if final:
    print("final")
    shap_select, explainer, model = shape_frame_all(new_first, new_second,target,node_size )
    ### Can be put into function, somewhat unnecessary
    shap_select_abs = shap_select[0].abs()
    shap_select_abs = shap_select_abs.sort_values(ascending=False)
    final_ft = shap_select_abs[shap_select_abs.cumsum().sub((shap_select_abs.sum()*final_contribution)).le(0)]
    final_ft = list(final_ft.index)
    final_ft.append(target)
    return new_first,new_second, mse,final_ft, shap_select_abs

  return new_first,new_second, mse


def all(new_first_two,new_second_two, shapper,target, dall=False ):
  def scaler(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
    return df

  if dall:
    shapper = new_first_two.columns
  

  new_first_y = new_first_two[target].copy()
  new_first = new_first_two[shapper].copy()
  new_first = scaler(new_first.drop([target],axis=1)).copy()

  new_second_y = new_second_two[target].copy()
  new_second = new_second_two[shapper].copy()
  new_second = scaler(new_second.drop([target],axis=1)).copy()

  
   

  return new_first,new_first_y, new_second, new_second_y


## Definitely a benefit to run twice but not three times. 
def runner(first_run, second_run, target, inter_portion=0.8, sqr_portion=0.9, contribution_portion=0.9, final_contribution=0.95, deflator=0.7, node_size=1.0 , runs=2, modeller="Keras"):
#def runner(first_run, second_run, target, inter_portion=0.4, sqr_portion=0.3, contribution_portion=0.9, deflator=0.6, runs=3):
  for r in range(runs):
    r += 1
    if r ==runs:
      print("final")
      first_run, second_run, mse, shapper, shap_select_abs = modelling(first_run, second_run, target, inter_portion, sqr_portion, contribution_portion, final_contribution,node_size, True, modeller)
    else:
      first_run, second_run, mse = modelling(first_run, second_run, target, inter_portion, sqr_portion, contribution_portion, final_contribution, node_size, False, modeller)

    inter_portion = inter_portion * deflator
    sqr_portion = sqr_portion * deflator * 1.1
    contribution_portion = contribution_portion * deflator
    gc.collect()


  new_deep_one,y_train, new_deep_two, y_test = all(first_run, second_run, shapper,target , True)

  new_first = new_deep_one.copy()
  new_second = new_deep_two.copy()
  lass = linear_model.LassoLarsCV(cv=5).fit(new_first, y_train)

  preds = lass.predict(new_second)
  mse = mean_squared_error(y_test, preds)
  print(mse)

  model = SelectFromModel(lass, prefit=True)
  X_train = new_first[new_first.columns[model.get_support(indices=True)].copy()].copy()
  X_test = new_second[list(X_train.columns)].copy()

  return X_train, y_train, X_test, y_test