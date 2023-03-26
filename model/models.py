from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



class all_models:
    
    def __init__(self):
        pass

    def linear_model(self,x,y):

        model = LinearRegression()
        model.fit(x,y)

        return model.coef_
    
    def polynomial_model_train(self, degree, x, y):

        poly_features = PolynomialFeatures(degree=degree,include_bias=False)
        X_poly_train = poly_features.fit_transform(x)

        polymodel = LinearRegression()
        # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=32)
        polymodel.fit(X_poly_train,y)

        x3 =polymodel.coef_[2]
        x2 =polymodel.coef_[1]
        x1 =polymodel.coef_[0]

        intercept = polymodel.intercept_
        # y_test_pred = model.predict(X_poly_train)

        return x3,x2,x1,intercept,polymodel
    
    def predict(self,model,x,degree):

        poly_features = PolynomialFeatures(degree=degree,include_bias=False)
        X_poly_test = poly_features.fit_transform(x)

        y_pred =  model.predict(X_poly_test)

        return y_pred