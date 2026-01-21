import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

st.title("AI 책상 클린 가이드: 객체 기반 강조")

def transform_and_detect(img):
    # 1. 기초 처리
    h, w = img.shape[:2]
    
    # 2. (가정) 객체 감지된 영역 설정 (YOLO boxes[i] 데이터를 기반으로 설정 가능)
    # 여기서는 예시로 화면 중앙의 책상 영역을 사각형으로 지정합니다.
    pts1 = np.float32([[w*0.2, h*0.3], [w*0.8, h*0.3], [w*0.1, h*0.9], [w*0.9, h*0.9]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # 3. Perspective Transform (원근 변환)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(img, matrix, (w, h))

    # 4. 더러운 부분(연필 자국) 강조 - Canny Edge
    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 5. 강조된 선을 원본 색상 위에 덧씌우기 (빨간색 선으로 표시)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[np.where((edges_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    
    result = cv2.addWeighted(transformed, 0.7, edges_colored, 0.3, 0)
    return result

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = transform_and_detect(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(key="ai-transform", video_frame_callback=video_frame_callback)
