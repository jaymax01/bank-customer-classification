import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

matplotlib.use('Agg')
st.set_option('deprecation.showPyplotGlobalUse', False)
import warnings
warnings.filterwarnings('ignore')


from PIL import Image

#Set title

st.title('Bank customers conversion rate')
image = Image.open('photo.png')
st.image(image,use_column_width=True)
st.subheader("Exploring the result of the result of Thera Bank's marketing campaign towards accepting its private loan", divider='orange')

def main():
        
    df = pd.read_csv('original.csv')
    
    activities=['Background info','EDA', 'Visualization','Model','Conclusion']
    option=st.sidebar.selectbox('Select option',activities)

    if option == 'Background info':
      st.subheader("Background Information")
      st.write("This project is about Thera Bank whose management wants to explore ways of converting its liability customers to personal loan customers while retaining them as depositors.")
      st.write("A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success.") 
      st.write("This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.")
      st.write("The goal is to compare the performance of classification algorithms (Logistic, naive bayes and KNN- k nearest neighbor) to predict or classify a customer as purchasing a personal loan or not.")
      st.write("The dataset for the project contains data on 5000 customers. The data include customer demographic information (age, income, etc.),")
      st.write("the customers relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan).")
      st.write("Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.")
      st.markdown('Below is the information about the data columns')
      st.write("ID - unique ID of the customer")
      st.write("Age - The age of the customer")
      st.write("Experience - work experience of the customer")
      st.write("Income - The income of the customer")
      st.write("Zip Code - The customer's zip code")
      st.write("Family - The family size of the customer")
      st.write("CCAvg - The average credit card spending of the customer")
      st.write("Education - The educational level of the customer")
      st.write("Mortgage - Indicator of if the customer has or does not have a mortgage")
      st.write("Personal loan - Indicates whether a customer purchased the private loan during the last campaign")
      st.write("Security Account - Indicates if the customer has a security account with the bank")
      st.write("CD Account - Indicates if the customer has a CD account with the bank")
      st.write("Online - Indicates if the customer banks online")
      st.write("CreditCard - Indicates if the customer has a credit card account with the bank")
      st.write()
      st.markdown('You can use this app from your classification tasks as well by uploading your data with a column that categorizes each data record')
      
    elif option == 'EDA':
        st.subheader("Exploritory Data Analysis")
        st.write('You can choose columns from the bank dataset below or upload your data to explore the data')

        data_upload = st.file_uploader('Upload a csv or exel file', )
        if data_upload is not None:
           df = pd.read_csv(data_upload)
                
        st.dataframe(df.head(50))
        
        if st.checkbox("Display shape"):
          st.write(df.shape)

        if st.checkbox("Display columns"):
          all_columns=df.columns.to_list()
          st.write(all_columns)

        if st.checkbox('Select Multiple columns'):
          selected_columns=st.multiselect("Select Preferred columns",df.columns)
          df1=df[selected_columns]
          st.dataframe(df1)

        if st.checkbox("Display summary"):
          df1=df[selected_columns]
          st.write(df1.describe().T)

        if st.checkbox("Null Values"):
          df1=df[selected_columns]
          st.write(df1.isnull().sum())

        if st.checkbox('Data Types'):
          df1=df[selected_columns]
          st.write(df1.dtypes)

        if st.checkbox('Display Correlation'):
          df1=df[selected_columns]
          plt.figure(figsize=(16,12))
          st.write(sns.heatmap(df1.corr(), vmax=1, annot=True, square=False))
          st.pyplot()

    elif option == 'Visualization':
        st.subheader("Visualization")
        
        data_upload = st.file_uploader('Upload a csv or exel file', )
        if data_upload is not None:
           df = pd.read_csv(data_upload)

        st.dataframe(df.head())

               
        if st.checkbox('Select Multiple Columns to plot'):
          selected_cols=st.multiselect("Select Your Preferred columns",df.columns)
          plots=st.selectbox("select a choice of plot",['histogram','bargraph','area plot','line plot'])
          df1=df[selected_cols]
          st.dataframe(df1)
          
        if st.button("Create Plot"):
          
          if plots=="area plot":
              df2=df[selected_cols]
              st.area_chart(df2)
              # st.write(df2.plot.area())
              st.success("success")
              st.pyplot()

          elif plots=="histogram":
              df2=df[selected_cols]
              st.write(plt.hist(df2, bins=20))
              st.success("success")
              st.pyplot()


          elif plots=="bargraph":
            df2=df[selected_cols]
            st.bar_chart(df2)
            st.success("success")
            st.pyplot()

          elif plots=="line plot":
            df2=df[selected_cols]
            st.line_chart(df2)
            st.success("success")
            st.pyplot()

        if st.checkbox("Display Heatmap"):
          plt.figure(figsize=(16,12))
          st.write(sns.heatmap(df.corr(), vmax=1, square=False, annot=True,cmap='viridis'))
          st.pyplot()

        if st.checkbox("Display Pairplot"):
          st.write(sns.pairplot(df,diag_kind='kde'))
          st.pyplot()

        if st.checkbox("Pie Chart"):
          all_columns=df.columns.to_list()
          pie_columns=st.selectbox("Select a column, NB: Select Target column",all_columns)
          pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
          st.write(pieChart)
          st.pyplot()
        
        
    elif option == 'Model':
        st.subheader("Model Building")
        data_upload = st.file_uploader('Upload a csv or exel file', )
        if data_upload is not None:
           df = pd.read_csv(data_upload)

        if st.checkbox('Select Multiple columns'):
            new_data=st.multiselect("Select column. NB: Make Sure Target column is selected last",df.columns)
            df1=df[new_data]
            st.dataframe(df1)
            X=df1.iloc[:,0:-1]
            y=df1.iloc[:,-1]


        seed = st.sidebar.slider('Seed for Data split', 1, 200)

        classifier_name = st.sidebar.selectbox('Select classifier',('KNN', 'SVM','LR','naive_bayes','DecisionTree', 'Xgb_classifier'))

      
        def add_parameter(name_of_clf):
            params = dict()
            if name_of_clf == 'SVM':
                C = st.sidebar.slider('C', 0.01, 15.0)
                params['C'] = C
            elif name_of_clf == 'KNN':
                K = st.sidebar.slider('K', 1, 15)
                S = st.sidebar.selectbox('Select an Algorithm', ['auto', 'ball_tree', 'kd_tree'])
                P = st.sidebar.slider('P', 1, 5)
                W = st.sidebar.selectbox('Select weights', ['uniform', 'distance'])
                M = st.sidebar.selectbox('Select metric', ['minkowski', 'euclidean', 'manhattan'])
                params['K'] = K
                params['S'] = S
                params['P'] = P
                params['W'] = W
                params['M'] = M
            else:
                N = st.sidebar.slider('Number of estimators', 1, 15)
                D = st.sidebar.slider('Maximum Dept', 1, 10)
                L = st.sidebar.slider('Learning Rate', min_value=0.0, max_value=1.0, step=0.01, format='%f')
                params['N'] = N
                params['D'] = D
                params['L'] = L
               
            return params

      #calling our function
        params = add_parameter(classifier_name)


        #accessing our classifier

        def get_classifier(name_of_clf, params):
            clf = None
            if name_of_clf == 'SVM':
                clf = SVC(C=params['C'])
            elif name_of_clf == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'], algorithm=params['S'], p=params['P'], weights=params['W'], metric=params['M'])
            elif name_of_clf=='LR':
              clf=LogisticRegression()
            elif name_of_clf=='naive_bayes':
              clf=GaussianNB()
            elif name_of_clf=='DecisionTree':
              clf=DecisionTreeClassifier()
            elif name_of_clf == 'Xgb_classifier':
              clf=XGBClassifier(n_estimators=params['N'], max_depth=params['D'], learning_rate=params['L'], objective='binary:logistic')

            else:
                st.warning('select your choice of algorithm')
            return clf


        clf = get_classifier(classifier_name, params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        
        smt = SMOTE(random_state=0)
        X_train, y_train = smt.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        scaler_fit = scaler.fit(X_train)
        X_train = scaler_fit.transform(X_train)
        X_test = scaler_fit.transform(X_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.write('Predictions',pd.concat([pd.DataFrame(data=scaler.inverse_transform(X_test), columns=new_data[:-1]), pd.Series(y_pred, name='predicted')], axis=1))
        st.write('Features importance for your chosen algorithm')

        def feature_importance(classifier_name=None):
           if classifier_name == 'Xgb_classifier' or classifier_name == 'DecisionTree':
              return st.write(pd.Series(clf.feature_importances_, index=new_data[:-1]))
           elif classifier_name == 'LR':
              return st.write(pd.Series(clf.coef_[0], index=new_data[:-1]))
           elif classifier_name == 'KNN':
              results = permutation_importance(clf, X_test, y_test, scoring='neg_mean_squared_error')
              return st.write(pd.Series(results.importances_mean, index=new_data[:-1]))
           elif classifier_name == 'SVM':
              results = permutation_importance(clf, X_test, y_test, scoring='neg_mean_squared_error')
              return st.write(pd.Series(results.importances_mean, index=new_data[:-1]))
           elif classifier_name == 'naive_bayes':
              results = permutation_importance(clf, X_test, y_test, scoring='neg_mean_squared_error')
              return st.write(pd.Series(results.importances_mean, index=new_data[:-1]))

        feature_importance(classifier_name)
        
        accuracy = accuracy_score(y_test, y_pred)

        st.write('Classifier name:',classifier_name)
        st.write('Accuracy:', accuracy)

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]], 
                                columns = [i for i in ["Predict 1","Predict 0"]])
        st.write('Confusion Matrix with the number of True and False Classifications')
        st.write(df_cm)

        

    elif option == 'Conclusion':
          st.subheader("Conclusion")
          st.write('The best classifiers to classify the customers into accepting the bank loan or not is the Xgboost classifier and the K nearest neighbor classifier')
          st.write('Both give accuracy scores of more than 90%. For the next ad campaign the company could experiment by picking the customer features that gives a high accuracy rate and use those to improve their customers conversion rates')




if __name__ == '__main__':
    main()