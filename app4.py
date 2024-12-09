# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
from openai import OpenAI
import os
# CSV 파일 로드
file_path = '/content/drive/MyDrive/인공지능기초/이자민/full_petition_df.csv'
data = pd.read_csv(file_path)

#사진 파일 로드
from PIL import Image

#PIL 패키지에 이미지 모듈을 통해 이미지 열기 
# Image.open('이미지 경로')
assembly_img = Image.open('/content/drive/MyDrive/인공지능기초/이자민/assembly_img.jpg')

col1,col2 = st.columns([2,3])
# 제목
st.image(assembly_img)
st.write('사진 출처: 대한민국 국회')
st.title("국회 국민동의 청원 대시보드")

# Tab 설정
tab1, tab2 = st.tabs(['국회 국민동의 청원 챗봇', '국회 국민동의 청원 현황'])

with tab1:
    # Tab A 내용
    #st.write('')
  
    os.environ["OPENAI_API_KEY"] = "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"  
    client = OpenAI()  
  
    st.title("당신의 청원 작성을 도와드립니다!")  

    if 'messages' not in st.session_state:  
        st.session_state.messages = [{"role":"system","content":"당신은 국회 국민동의 청원의 주제를 요약하고, 정해진 18가지의 분류(1.정치/선거/국회운영 2.수사/법무/사법제도 3.재정/세재/금융/예산 4.소비자/공정거래 5.교육 6.과학기술/정보통신 7.외교/통일/국방/안보 8.재난/안전/환경 9.행정/지방자치 10.문화/체육/관광/언론 11.농업/임업/수산업/축산업 12.산업/통상 13.보건의료 14.복지/보훈 15.국토/해양/교통 16.인권/성평등/노동 17.저출산/고령화/아동/청소년/가족 18.기타)에 따라 청원을 분류해주는 인공지능이야. 또한 주제와 문제 상황을 설명해줬을때, 청원을 1000자 이상으로 완성시켜주는 일도 수행해야해. 사용자의 요구에 맞춰 적절한 응답을 내어줘. 청원 요약시에는 청원 텍스트 내에서 주제와 관련된 유의미한 단어만 사용해서 텍스트 버블 이미지를 만들어줘. 이때 사용되는 단어는 30단어를 넘지 말아. 너가 작성한 청원 전문이나, 입력된 청원 전문이든, 전문이 존재하기만 한다면 텍스트 버블 이미지를 항상 출력해줘."}
        ,{"role":"assistant","content":""}]  
  
    for msg in st.session_state.messages:
        if msg["role"] != "system":  # 시스템 메시지는 제외
            if msg["role"] == "assistant":
                st.write(f"🤖 {msg['content']}")
            elif msg["role"] == "user":
                st.write(f"🧑 {msg['content']}")

    user_message = st.text_input("User:", key="text_input1")  
    
    if st.button("Send"):
    # 사용자 메시지 추가
        if user_message.strip():
            st.session_state.messages.append({"role": "user", "content": user_message})
        
        # OpenAI 응답 생성
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages
        )
            response = completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})


# 메시지 출력 (system 메시지 제외, 중복 출력 방지)
    for msg in reversed(st.session_state.messages):
        if msg["role"] != "system":  # system 메시지 제외
            with st.chat_message(msg['role']):
                st.write(msg['content'])

with tab2:
    # Tab B 내용
    st.write('국회 국민동의 청원 대시보드')
    col1, col2 = st.columns([2, 3])

    # Column 1
    with col1:
        st.subheader("주제별 청원 분류")
        grouped_by_topic = data.groupby("분야").size()  # 청원 주제별 건수 집계
        fig1, ax1 = plt.subplots()
        ax1.pie(
            grouped_by_topic, 
            labels=grouped_by_topic.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.cm.Paired.colors
        )
        ax1.set_title("주제별 청원 분포")
        fig1.savefig('/content/topic.png')
        st.image('/content/topic.png')

    # Column 2
    with col2:
        st.subheader("청원 동의 수")
        grouped_by_agreements = data.groupby("분야")["동의수"].sum().sort_values(ascending=False)  # 주제별 동의 수 합산
        fig2, ax2 = plt.subplots()
        ax2.bar(
            grouped_by_agreements.index, 
            grouped_by_agreements.values, 
            color="skyblue"
        )
        ax2.set_title("청원 주제별 동의 수")
        ax2.set_ylabel("동의 수")
        ax2.set_xlabel("청원 주제")
        plt.xticks(rotation=45)
        fig2.savefig('/content/nums.png')
        st.image('/content/nums.png')

import os
import re
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 사용자 정의 전처리 함수
def custom_preprocessor(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

with st.sidebar:
    st.title("내 청원은 중요할까?")
    st.write('청원 중요도 계산기')
    st.write('계산기는 지역, 소수자성 등 동의수 기준의 국민 청원에서 무시될 수 있는 요소에 가중치를 부여합니다.')
    text_data = st.text_input("텍스트를 입력하세요:", key="text_input2")

# 파일 경로
    tokenizer1_path = "/content/drive/MyDrive/인공지능기초/이자민/local_weighted_tokenizer.pkl"
    tokenizer2_path = "/content/drive/MyDrive/인공지능기초/이자민/minority_weighted_tokenizer.pkl"
    model1_path = "/content/drive/MyDrive/인공지능기초/이자민/local_weighted_model.pkl"
    model2_path = "/content/drive/MyDrive/인공지능기초/이자민/minority_weighted_model.pkl"

# 전처리기 및 모델 로드
    try:
        tokenizer1 = AutoTokenizer.from_pretrained(tokenizer1_path)
        model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)

        tokenizer2 = AutoTokenizer.from_pretrained(tokenizer2_path)
        model2 = AutoModelForSequenceClassification.from_pretrained(model2_path)
    except Exception as e:
        st.error(f"토크나이저 또는 모델 로드 중 오류 발생: {e}")

    if text_data:
        try:
        # 사용자 입력 텍스트 토크나이징 (Hugging Face 방식)
            inputs1 = tokenizer1(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs2 = tokenizer2(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 모델 예측
            with torch.no_grad():
                outputs1 = model1(**inputs1)
                outputs2 = model2(**inputs2)

        # 로짓을 스칼라로 변환 (Softmax를 사용하여 확률로 변환 가능)
            prediction1_scalar = torch.softmax(outputs1.logits, dim=1).max().item()
            prediction2_scalar = torch.softmax(outputs2.logits, dim=1).max().item()
        
            final_predict = max(prediction1_scalar, prediction2_scalar)

        # 결과 출력
            if final_predict == prediction1_scalar:
                st.write(f"이 청원은 지역성을 지닙니다. 지역성에서 다음의 가중치를 얻습니다.: {prediction1_scalar:.2f}")
            else:
                st.write(f"이 청원은 소수자성을 지닙니다. 소수자성에서 다음의 가중치를 얻습니다.: {prediction2_scalar:.2f}")

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
    else:
        st.write("텍스트를 입력하세요.")