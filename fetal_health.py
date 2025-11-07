import numpy as np
import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import pickle
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def csv_upload(file):
    x = pd.read_csv(file)
    return x

fh_df = csv_upload('fetal_health.csv')
fh_df.drop(columns = 'fetal_health', inplace = True)

@st.cache_resource
def load_dt(file):
    dt_class = open(file, 'rb')
    x = pickle.load(dt_class)
    dt_class.close()
    return x
dt_ml = load_dt('dt_ml.pickle')

@st.cache_resource
def load_rf(file):
    rf_class = open(file, 'rb')
    x = pickle.load(rf_class)
    rf_class.close()
    return x
rf_ml = load_rf('rf_ml.pickle')

@st.cache_resource
def load_ada(file):
    ada_class = open(file, 'rb')
    x = pickle.load(ada_class)
    ada_class.close()
    return x
ada_ml = load_ada('ada_ml.pickle')

@st.cache_resource
def load_vc(file):
    vc_class = open(file, 'rb')
    x = pickle.load(vc_class)
    vc_class.close()
    return x
vc_ml = load_vc('vc_ml.pickle')

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif')
st.text('Use our advanced machine learning app to predict fetal health classifications')

st.sidebar.header('Fetal Health Feature Inputs')

file_upload = st.sidebar.file_uploader('Upload your data', help = 'File must be in CSV format')
st.sidebar.warning('Ensure your data strictly follows the format outlined below')
st.sidebar.dataframe(fh_df.head(), width = 'stretch')
model = st.sidebar.radio('Choose Model for Prediction', options = ['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])
st.sidebar.info(f'You selected: {model}')

if file_upload is None:
    st.info('Please upload data to proceed')
if file_upload is not None:
    st.success('CSV file uploaded successfully')
    st.subheader(f'Predicting Fetal Health Class Using {model} Model')
    file_df = pd.read_csv(file_upload)
    if model == 'Random Forest':
        predict = rf_ml.predict(file_df)
        probability = rf_ml.predict_proba(file_df).max(axis=1)
        predict_list = []
        color_list = []
        for i in predict:
            if i == 1:
                predict_list.append('Normal')
                color_list.append('Green')
            elif i == 2:
                predict_list.append('Suspect')
                color_list.append('#eddd2f')
            elif i == 3:
                predict_list.append('Pathological')
                color_list.append('Orange')
        file_df['Predicted Fetal Health'] = predict_list
        file_df['Prediction Probability(%)'] = (probability*100).round(2)
        def color_col(col):
            return pd.Series([f"background-color: {c}" for c in color_list], index = col.index)
        styled_df = file_df.style.apply(color_col, subset=['Predicted Fetal Health']).format({"Prediction Probability(%)": "{:.2f}%"})
        st.dataframe(styled_df, width = 'stretch')

        st.subheader('Model Performance and Insights')
        tab1, tab2, tab3 = st.tabs(['Confusion Matrix', 'Classification Report', 'Feature Importance'])
        with tab1:
            st.subheader('Confusion Matrix')
            st.image('rf_confusion_matrix.svg')
            st.caption('Confusion matrix for Random Forest Model Prediction')
        
        with tab2:
            st.subheader('Classification Report')
            rf_report = pd.read_csv('rf_class_report.csv', index_col = 0)
            rf_report.rename(columns={'1.0' : 'Normal', '2.0' : 'Suspect', '3.0' : 'Pathological'}, inplace = True)
            rf_report = rf_report.transpose()
            styled_rf_report = rf_report.style.background_gradient(cmap = 'PuBu')
            st.dataframe(styled_rf_report)
            st.caption('Classification Report for Random Forest Model')

        with tab3:
            st.subheader('Feature Importance Analysis')
            st.image('rf_feature_importance.svg')
            st.caption('Feature Importance Chart for Random Forest Model')

    elif model == 'Decision Tree':
            predict = dt_ml.predict(file_df)
            probability = dt_ml.predict_proba(file_df).max(axis=1)
            predict_list = []
            color_list = []
            for i in predict:
                if i == 1:
                    predict_list.append('Normal')
                    color_list.append('Green')
                elif i == 2:
                    predict_list.append('Suspect')
                    color_list.append('#eddd2f')
                elif i == 3:
                    predict_list.append('Pathological')
                    color_list.append('Orange')
            file_df['Predicted Fetal Health'] = predict_list
            file_df['Prediction Probability(%)'] = (probability*100).round(2)
            def color_col(col):
                return pd.Series([f"background-color: {c}" for c in color_list], index = col.index)
            styled_df = file_df.style.apply(color_col, subset=['Predicted Fetal Health']).format({"Prediction Probability(%)": "{:.2f}%"})
            st.dataframe(styled_df, width = 'stretch')

            st.subheader('Model Performance and Insights')
            tab1, tab2, tab3 = st.tabs(['Confusion Matrix', 'Classification Report', 'Feature Importance'])
            with tab1:
                st.subheader('Confusion Matrix')
                st.image('dt_confusion_matrix.svg')
                st.caption('Confusion matrix for Decision Tree Model Prediction')
            
            with tab2:
                st.subheader('Classification Report')
                dt_report = pd.read_csv('dt_class_report.csv', index_col = 0)
                dt_report.rename(columns={'1.0' : 'Normal', '2.0' : 'Suspect', '3.0' : 'Pathological'}, inplace = True)
                dt_report = dt_report.transpose()
                styled_dt_report = dt_report.style.background_gradient(cmap = 'YlGn')
                st.dataframe(styled_dt_report)
                st.caption('Classification Report for Decision Tree Model')

            with tab3:
                st.subheader('Feature Importance Analysis')
                st.image('dt_feature_importance.svg')
                st.caption('Feature Importance Chart for Decision Tree Model')

    elif model == 'AdaBoost':
        predict = ada_ml.predict(file_df)
        probability = ada_ml.predict_proba(file_df).max(axis=1)
        predict_list = []
        color_list = []
        for i in predict:
            if i == 1:
                predict_list.append('Normal')
                color_list.append('Green')
            elif i == 2:
                predict_list.append('Suspect')
                color_list.append('#eddd2f')
            elif i == 3:
                predict_list.append('Pathological')
                color_list.append('Orange')
        file_df['Predicted Fetal Health'] = predict_list
        file_df['Prediction Probability(%)'] = (probability*100).round(2)
        def color_col(col):
            return pd.Series([f"background-color: {c}" for c in color_list], index = col.index)
        styled_df = file_df.style.apply(color_col, subset=['Predicted Fetal Health']).format({"Prediction Probability(%)": "{:.2f}%"})
        st.dataframe(styled_df, width = 'stretch')

        st.subheader('Model Performance and Insights')
        tab1, tab2, tab3 = st.tabs(['Confusion Matrix', 'Classification Report', 'Feature Importance'])
        with tab1:
            st.subheader('Confusion Matrix')
            st.image('ada_confusion_matrix.svg')
            st.caption('Confusion matrix for AdaBoost Model Prediction')
        
        with tab2:
            st.subheader('Classification Report')
            ada_report = pd.read_csv('ada_class_report.csv', index_col = 0)
            ada_report.rename(columns={'1.0' : 'Normal', '2.0' : 'Suspect', '3.0' : 'Pathological'}, inplace = True)
            ada_report = ada_report.transpose()
            styled_ada_report = ada_report.style.background_gradient(cmap = 'RdPu')
            st.dataframe(styled_ada_report)
            st.caption('Classification Report for AdaBoost Model')

        with tab3:
            st.subheader('Feature Importance Analysis')
            st.image('ada_feature_importance.svg')
            st.caption('Feature Importance Chart for AdaBoost Model')

    elif model == 'Soft Voting':
        predict = vc_ml.predict(file_df)
        probability = vc_ml.predict_proba(file_df).max(axis=1)
        predict_list = []
        color_list = []
        for i in predict:
            if i == 1:
                predict_list.append('Normal')
                color_list.append('Green')
            elif i == 2:
                predict_list.append('Suspect')
                color_list.append('#eddd2f')
            elif i == 3:
                predict_list.append('Pathological')
                color_list.append('Orange')
        file_df['Predicted Fetal Health'] = predict_list
        file_df['Prediction Probability(%)'] = (probability*100).round(2)
        def color_col(col):
            return pd.Series([f"background-color: {c}" for c in color_list], index = col.index)
        styled_df = file_df.style.apply(color_col, subset=['Predicted Fetal Health']).format({"Prediction Probability(%)": "{:.2f}%"})
        st.dataframe(styled_df, width = 'stretch')

        st.subheader('Model Performance and Insights')
        tab1, tab2, tab3 = st.tabs(['Confusion Matrix', 'Classification Report', 'Feature Importance'])
        with tab1:
            st.subheader('Confusion Matrix')
            st.image('vc_confusion_matrix.svg')
            st.caption('Confusion matrix for Soft Voting Model Prediction')
        
        with tab2:
            st.subheader('Classification Report')
            vc_report = pd.read_csv('vc_class_report.csv', index_col = 0)
            vc_report.rename(columns={'1.0' : 'Normal', '2.0' : 'Suspect', '3.0' : 'Pathological'}, inplace = True)
            vc_report = vc_report.transpose()
            styled_vc_report = vc_report.style.background_gradient(cmap = 'OrRd')
            st.dataframe(styled_vc_report)
            st.caption('Classification Report for Soft Voting Model')

        with tab3:
            st.subheader('Feature Importance Analysis')
            st.image('vc_feature_importance.svg')
            st.caption('Feature Importance Chart for Soft Voting Model')

# For this app, I used CoPilot(AI) to help me with the conditional formatting for the background color in
# the dataframe.  It presented me with the solution of using the .style feature for the dataframe.  I also used
# AI for general debugging whenever I ran into an error.