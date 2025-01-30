import numpy as np

from sklearn.metrics import mean_squared_error

class BaseModel:

    def __init__(self,dataset,model,param_grid, is_multi_output=False, target_variable_name = None):
        
        self.dataset = dataset.copy()
        self.target_variable_name = target_variable_name

        if self.target_variable_name:
            self.dataset.Y_train = self.dataset.Y_train[self.target_variable_name]
            self.dataset.Y_val = self.dataset.Y_val[self.target_variable_name]
            self.dataset.Y_test = self.dataset.Y_test[self.target_variable_name]

        self.model = model
        self.is_multi_output = is_multi_output
        self.param_grid = param_grid
        self.best_params = None

        self.random_state = 42

    
    def computeRMSE(self, Y,Y_hat):
        """
        Just a method to compute RMSE
        """
        
        if self.is_multi_output:
            n_value_rmse,sc_value_rmse,st_value_rmse = np.sqrt(mean_squared_error(y_true=Y,y_pred=Y_hat, multioutput='raw_values'))
            return n_value_rmse, sc_value_rmse, st_value_rmse
        
        return np.sqrt(mean_squared_error(y_true=Y,y_pred=Y_hat))


    def CrossValidate(self,x_train,y_train,x_val,y_val):
        self.model.fit(x_train,y_train)
        y_hat = self.model.predict(x_val)

        if self.is_multi_output:
            n_value_rmse,sc_value_rmse,st_value_rmse = self.computeRMSE(y_val, y_hat)
            return n_value_rmse, sc_value_rmse, st_value_rmse
        
        return  self.computeRMSE(y_val, y_hat)
    
            
    def validate(self):
        """
        Can be used for Hyperparameter Tuning.
        """
        self.model.fit(self.dataset.X_train,self.dataset.Y_train)
        y_hat = self.model.predict(self.dataset.X_val).squeeze()
        
        return  self.computeRMSE(self.dataset.Y_val, y_hat)
    
    def evaluate(self):
        """
        Use this for testing only! (After training the model using CV10)
        """
        y_hat = self.model.predict(self.dataset.X_test).squeeze()

        return self.computeRMSE(self.dataset.Y_test,y_hat)











        

            



