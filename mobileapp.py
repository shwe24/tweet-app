import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile

import os
os.system( '7z x mobile.7z' )

st.title("Sentiment analysis of Mobile phone brands")
st.sidebar.title("Customer Satisfaction Reviews on Mobile brands")

data_link=("mobile.csv")
@st.cache()
def load_data():
    df=pd.read_csv(data_link)
    senti={1:'Negative',2:'Negative',3:'Neutral',4:'Positive',5:'Positive'}
    df['Sentiment']=df['Rating'].map(senti)
    df=df.dropna()
    return df

df = load_data()

alg_data=("algo.csv")
@st.cache()
def load_data_alg():
    df1=pd.read_csv(alg_data)
    return df1

alg = load_data_alg()


##st.sidebar.markdown(
##        f"<style> body{{ background-color: #FFE4E1;}}</style>",    
##        unsafe_allow_html=True
##    )
brand=df['Brand Name'].value_counts().index.tolist()
list_brand=brand[1:50]

nav=st.sidebar.radio('Tabs: ',('Customer reviews','Sentiment Analysis'))

if nav=="Customer reviews":
    st.subheader("Customer reviews : Random Reviews")
    brand=st.selectbox('Brand name',(list_brand))
    random_tweet=st.radio('Sentiment',('Positive','Neutral','Negative'))
    choice=df[df['Brand Name']==brand]
    st.subheader("Random text:")
    st.write(choice.query('Sentiment==@random_tweet')[['Reviews']].sample(n=1).iat[0,0])

    st.write(f'**Number of tweets by Sentiment for {brand} **')
    sent_count=choice['Sentiment'].value_counts()
    sent_count=pd.DataFrame({'Sentiment':sent_count.index, 'Tweets':sent_count.values})
    fig=px.bar(sent_count,x='Sentiment',y='Tweets', color='Tweets')
    st.plotly_chart(fig)
    numb=choice['Reviews'].count()
    avg_rating=choice['Rating'].mean()
    st.sidebar.markdown(f'<p style="font-size:30px"><b>Brand: {brand}</b></p>',unsafe_allow_html=True)
    if round(avg_rating)==5:
        st.sidebar.markdown(f'<center><h1>Overall Ratings</center><h1><p style="font-size:50px; color:green"><b><center>{round(avg_rating,1)}</center></b></p><center><img src="https://www.mortgagebrokergoldcoast.com/wp-content/uploads/2020/03/5-stars.png width=250 height=60"></img></center><center><p style="font-size:50px; color:green">Excellent</center></p></h1>', unsafe_allow_html=True)
    elif round(avg_rating)==4:
        st.sidebar.markdown(f'<center><h1>Overall Ratings</center><h1><p style="font-size:50px; color:green"><b><center>{round(avg_rating,1)}</center></b></p><center><img src="https://bitls.yolasite.com/resources/images/misc/4-Stars.jpg" width=250 height=60></img></center><center><p style="font-size:50px; color:green">Good</center></p></h1>', unsafe_allow_html=True)
    elif round(avg_rating)==3:
        st.sidebar.markdown(f'<center><h1>Overall Ratings</center><h1><p style="font-size:50px; color:orange"><b><center>{round(avg_rating,1)}</center></b></p><center><img src="https://png.pngitem.com/pimgs/s/398-3984464_film-star-rating-three-stars-transparent-background-5.png" width=250 height=100></img></center><center><p style="font-size:50px; color:yellow">Not bad</center></p></h1>', unsafe_allow_html=True)
    elif round(avg_rating)==2:
        st.sidebar.markdown(f'<center><h1>Overall Ratings</center><h1><p style="font-size:50px; color:red"><b><center>{round(avg_rating,1)}</center></b></p><center><img src="https://www.nhtsa.gov/sites/nhtsa.dot.gov/themes/nhtsa_gov/images/star-rating/2.png" width=250 height=60></img></center><center><p style="font-size:50px; color:red">Bad</center></p> </h1>',unsafe_allow_html=True)

    else:
        st.sidebar.markdown(f'<center><h1>Overall Ratings</center><h1><p style="font-size:50px; color:red"><b><center>{round(avg_rating,1)}</center></b></p><center><img src="https://wiki.starwarsminute.com/images/7/7c/1-star.jpg" width=250 height=60></img></center><center><p style="font-size:50px; color:red">Poor</center></p> </h1>',unsafe_allow_html=True)

    st.sidebar.markdown(f' <p style="font-size: 25px"><center><b> {numb} tweets </b></center></p>', unsafe_allow_html=True)

if nav=="Sentiment Analysis":
    st.sidebar.subheader("Comparing classifiers")
    choice=st.sidebar.multiselect('Pick algorithm',('Naive Bayes','SVM','Bagging','Logistic Regression','Decision tree','KNN'))
    if len(choice)>0:
        choice_data=alg[alg.Algorithm.isin(choice)]
        fig_choice=px.histogram(choice_data,x='Algorithm',y='Accuracy',color='category', facet_col='category',height=600, width=800,labels={'Algorithm':'Algorithms','Accuracy':'Accuracy'})
        fig_choice1=px.histogram(choice_data,x='Algorithm',y='F1-score',color='category', facet_col='category',height=600, width=800,labels={'Algorithm':'Algorithms','F1-score':'F1-score'})
        st.subheader("Accuracy of classifiers")
        st.plotly_chart(fig_choice)
        st.subheader("F1-score of classifiers")
        st.plotly_chart(fig_choice1)

hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
