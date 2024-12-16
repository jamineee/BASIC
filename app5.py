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
file_path = './full_petition_df.csv'
data = pd.read_csv(file_path)

#사진 파일 로드
from PIL import Image

#PIL 패키지에 이미지 모듈을 통해 이미지 열기 
assembly_img = Image.open('./assembly_img.jpg')

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
  
    import os
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]    
    client = OpenAI()  

    st.title("당신의 청원 작성을 도와드립니다!")  

    if 'messages' not in st.session_state:  
        st.session_state.messages = [{"role":"system","content":"당신은 국회 국민동의 청원의 주제를 요약하고, 정해진 18가지의 분류(1.정치/선거/국회운영 2.수사/법무/사법제도 3.재정/세재/금융/예산 4.소비자/공정거래 5.교육 6.과학기술/정보통신 7.외교/통일/국방/안보 8.재난/안전/환경 9.행정/지방자치 10.문화/체육/관광/언론 11.농업/임업/수산업/축산업 12.산업/통상 13.보건의료 14.복지/보훈 15.국토/해양/교통 16.인권/성평등/노동 17.저출산/고령화/아동/청소년/가족 18.기타)에 따라 청원을 분류해주는 인공지능이야. 또한 주제와 문제 상황을 설명해줬을때, 청원을 1000자 이상으로 완성시켜주는 일도 수행해야해. 사용자의 요구에 맞춰 적절한 응답을 내어줘. 청원 요약시에는 청원 텍스트 내에서 주제와 관련된 유의미한 단어만 사용해서 텍스트 버블 이미지를 만들어줘. 이때 사용되는 단어는 30단어를 넘지 말아. 너가 작성한 청원 전문이나, 입력된 청원 전문이든, 전문이 존재하기만 한다면 텍스트 버블 이미지를 항상 출력해줘. 모든 출력은 다음의 형식을 따라서 작성해줘. (형식: 1. 주제 요약 2. 국회국민동의 청원 기반 분류 3. 청원 요약 시각화 이미지-워드클라우드 - 가장 핵심적인 단어 15개만 사용해서 그려줘. 4. 청원 작성을 요구했을경우, 청원을 작성한 후에 3번 작업을 진행해줘)"}
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
        topic_img = Image.open('./topic_agreements.png')
        st.image(topic_img)


    # Column 2
    with col2:
        st.subheader("청원 동의 수")
        agree_img = Image.open('./topic_distribution.png')
        st.image(agree_img)
