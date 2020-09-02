import pickle
import pandas as pd
import streamlit as st
import warnings; warnings.simplefilter('ignore')
import numpy as np

pickle_in_model = open("voting_clf.pkl", "rb")
classifier = pickle.load(pickle_in_model)

pickle_in_pipeline = open("full_pipeline.pkl", "rb")
full_pipeline = pickle.load(pickle_in_pipeline)

# loading your processed data for prediction
df = pd.read_excel("data_for_prediction.xlsx")
df["Pers No"] = df["Pers No"].astype(str)


def predict_performace(employee_id):
    
    to_predict = df[df["Pers No"]==employee_id].iloc[:,6:]
    to_predict_prep = full_pipeline.transform(to_predict)
   
    prediction=classifier.predict(to_predict_prep)
    
    print(prediction)
    return prediction


data_to_show = pd.DataFrame(data=None, columns=['Pers No', 'Employee Name', 
                                      'Position', 'Directorate', 'Department',
                                      'Unit', 'Prediction'])

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1400px;
        padding-top: 5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 10rem;
    }}
</style>
""",
        unsafe_allow_html=True,)


st.title("HC Analytics")
html_temp = """
    <div style="background-color:#f0b837;padding:10px">
    <h2 style="color:white;text-align:center;"><b>Employee Performance Prediction for Job Rotation (Prototype)</b></h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


employee_id = st.text_input("","Enter Employee ID")

@st.cache(allow_output_mutation=True)
def get_data():
    return []


result=""
if st.button("PREDICT"):
    
    result=predict_performace(employee_id)[0]
    st.success('The Predicted Performance is {}'.format(result))

    emp_info_results = df[df["Pers No"]==employee_id].iloc[:,:5]
    emp_info_results["Previous PA"] = df[df["Pers No"]==employee_id].iloc[:,11]
    emp_info_results["Prediction"] = result
    dictio = emp_info_results.to_dict(orient='records')[0]
    get_data().append(dictio)
    
    tr = emp_info_results.T.reset_index()
    tr.columns = ["Info","Details"]
    tr.index = np.arange(1,len(tr)+1)
    tr.index.name = "No"
    st.write(tr)    

    data = pd.DataFrame(get_data())
    data.index = np.arange(1,len(data)+1)
    data.index.name = "No"
    st.write(data)
