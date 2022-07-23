import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_player import st_player
from pipeline import input_func, helper_func
import streamlit.components.v1 as components
transformer = joblib.load('transformer.sav')
pca = joblib.load('pca.sav')
model = joblib.load('model.sav')
st.set_page_config(layout="wide")
with st.sidebar:
    
    choose = option_menu("Welcome", ["Home", "Tech Stack","Predictor","ML Code", "Contributors"],
                         icons=['house', 'stack', 'cpu','code-slash', 'people-fill'],
                         menu_icon="bank", default_index=0, 
                         
                         styles={
                            "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                            "icon": {"color": "White", "font-size": "25px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4d4d4d"},
                            "nav-link-selected": {"background-color": "#4d4d4d"},
                        }
    ) 


with open("contributors.html",'r') as f:
   contributors=f.read();
def html():
    components.html(
      contributors
     ,
    height=1400,
    
    scrolling=True,
)
def pred():
    st.title("CREDIT APPROVAL SYSTEM")
    
    BankCustomer = st.selectbox("         ", ('Yes', 'No'))
    st.write('Is the client an existing bank customer ?')
    if BankCustomer=='Yes':
        BankCustomer=1
    else:
        BankCustomer=0
    
    
    
    PriorDefault = st.selectbox("   ", ('Yes', 'No'))
    st.write('Has the client defaulted previously ?')
    if PriorDefault=='Yes':
        PriorDefault=1
    else:
        PriorDefault=0
    
    
    Employed = st.selectbox("       ", ('Yes', 'No'))
    st.write('Is the client currently employed ?')
    if Employed=='Yes':
        Employed=1
    else:
        Employed=0
    
    
    DriversLicense = st.selectbox(" ", ('Yes', 'No'))
    st.write('Does the client have a driving license ?')
    if DriversLicense=='Yes':
        DriversLicense=1
    else:
        DriversLicense=0
    
    Industry =  st.selectbox(' ',('Industrials', 'Materials', 'CommunicationServices', 'Transport',
       'InformationTechnology', 'Financials', 'Energy', 'Real Estate',
       'Utilities', 'ConsumerDiscretionary', 'Education',
       'ConsumerStaples', 'Healthcare', 'Research'))
    st.write("Client's industry of work")

    YearsEmployed = st.number_input('',max_value=40,min_value=0)
    st.write('Number of years of employment')


    
    Age = st.number_input('',max_value=90,min_value=18)
    st.write('Age')

    
    CreditScore = st.number_input('',max_value=31 , min_value=0)
    st.write("Client's credit score")

    
    Debt = st.number_input('',max_value=30,min_value=0)
    st.write('Total debt of the client')
    
    Income = st.number_input('',max_value=99999999999999,min_value=0)
    st.write('Annual income of the client')
    st.write("")
    df = pd.DataFrame({'Age':[Age], 'Debt': [Debt], 'BankCustomer': [BankCustomer], 'Industry':[Industry], 'YearsEmployed':[YearsEmployed], 'PriorDefault':[PriorDefault], 'Employed':[Employed], 'CreditScore':[CreditScore], 'DriversLicense':[DriversLicense], 'Income':[Income]})

    df = helper_func(df)

    df = transformer.transform(df)
    df = pca.transform(df)
    ypred = model.predict(df)
    if(st.button("Submit")):
        ans = bool((ypred[0]))
        if ans:
            st.success("Credit Approved ")
        else:
            st.error("Credit Rejected.")

with open('techstack.html','r') as f:
  techstack=f.read();
def tech():
    components.html(
    techstack
    ,
    height=1000,
    
    scrolling=True,
    )
def ml():
  st.write("To view the complete code for the end-to-end project, visit our [GitHub](https://github.com/snshahgit/credit-approval-system)")
  components.iframe("https://www.kaggle.com/embed/sns5154/credit-approval-system-val-87-50-test-81-16?kernelSessionId=100312408",height=1000,)





if choose=="Predictor":

    pred()
elif choose=="Home":
    st.title('AI for FinTech')

    st.write('')
    st.subheader("What's the need ?")
    st.markdown("<p style='text-align: justify;'>The Credit Approval procedures have a lot of human level interference. A client's application might be discriminated based on gender, race, caste, creed, and religion. Evidently, there is a pressing need for ethically and socially responsible AI solutions.</p>", unsafe_allow_html=True)
    st.write('')
    st.write('')

    st.subheader("How does this work ?")
    st.markdown("<p style='text-align: justify;'>This project diagnostically predicts the credit aplication approval of bank client. Client data used to train our Machine Learning model includes data such as employment information, previous defaults, industry of work, age, credit score, debt, and income.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>We have made sure that our model doesn't learn sexism, racism, and other forms of discrimination based on locality (zipcode), ethnicity, and last name (religion).</p>", unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center;'>Healthcare AI</h1>", unsafe_allow_html=True)

    # with open("pic.html",'r') as f:
    #     pic=f.read();
    # components.html(pic, height=400)

    # def load_lottieurl(url: str):
    #     r = requests.get(url)
    #     if r.status_code != 200:
    #         return None
    #     return r.json()
 
    # lt_url_hello = "https://assets6.lottiefiles.com/packages/lf20_1yy002na.json"
    # lottie_hello = load_lottieurl(lt_url_hello)
 
    # st_lottie(
    #         lottie_hello,  
    #         key="hello",
    #         speed=1,
    #         reverse=False,
    #         loop=True,
    #         quality="low",
    #         height=400,
    #         width=400            
    # )

    
elif choose=="Tech Stack":
    tech()
elif choose=="Contributors":
    html()
elif choose=="ML Code":
  ml()
