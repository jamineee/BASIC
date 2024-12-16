# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
import openai  # OpenAI 라이브러리 사용
import os

# CSV 파일 로드
file_path = './full_petition_df.csv'
data = pd.read_csv(file_path)

# 이미지 로드
assembly_img = Image.open('./assembly_img.jpg')
topic_img = Image.open('./topic_agreements.png')
agree_img = Image.open('./topic_distribution.png')

# Streamlit 제목 및 이미지
st.image(assembly_img)
st.write('사진 출처: 대한민국 국회')
st.title("국회 국민동의 청원 대시보드")

# Tab 설정
tab1, tab2 = st.tabs(['국회 국민동의 청원 챗봇', '국회 국민동의 청원 현황'])

with tab1:
    # OpenAI API 키 설정
    try:
        openai.api_key = st.secrets["openai"]["api_key"]  # Secrets에서 키 가져오기
    except KeyError:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")

    st.title("당신의 청원 작성을 도와드립니다!")

    # 메시지 관리
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "당신은 국회 국민동의 청원의 주제를 요약하고..."}]

    # 이전 메시지 출력
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.write(f"🤖 {msg['content']}")
        elif msg["role"] == "user":
            st.write(f"🧑 {msg['content']}")

    # 사용자 입력
    user_message = st.text_input("User:", key="text_input1")
    if st.button("Send"):
        if user_message.strip():
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": user_message})
            # OpenAI API 호출
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=st.session_state.messages
                )
                response = completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"OpenAI API 호출 중 에러 발생: {e}")

    # 채팅 메시지 출력
    for msg in reversed(st.session_state.messages):
        if msg["role"] != "system":
            st.write(f"{msg['role']}: {msg['content']}")

with tab2:
    # 대시보드 내용
    st.write('국회 국민동의 청원 대시보드')

    st.subheader("주제별 청원 분류")
    st.image(topic_img)

    st.subheader("청원 동의 수")
    st.image(agree_img)
