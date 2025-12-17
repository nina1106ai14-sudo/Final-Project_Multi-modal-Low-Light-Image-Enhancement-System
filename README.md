# 🌑 Multi-modal Low-Light Image Enhancement System
# 基於多模態融合與全自適應邏輯的低光影像增強系統

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow?logo=ultralytics)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **"Turning Darkness into Vision"** > 一個不依賴深度學習黑盒子，採用可解釋性電腦視覺演算法 (Explainable CV) 打造的智慧夜視增強系統。

---

## 📖 目錄 (Table of Contents)
- [專案簡介 (Introduction)](#-專案簡介-introduction)
- [核心亮點 (Key Features)](#-核心亮點-key-features)
- [系統架構與演算法 (System Architecture)](#-系統架構與演算法-system-architecture)
    - [1. 光照模態 (Modality A)](#1-光照模態-illumination-modality)
    - [2. 結構模態 (Modality B)](#2-結構模態-structure-modality)
    - [3. 自適應融合 (Adaptive Fusion)](#3-自適應融合-adaptive-fusion)
- [安裝與環境設定 (Installation)](#-安裝與環境設定-installation)
- [使用指南 (Usage Guide)](#-使用指南-usage-guide)
    - [GUI 操作說明](#gui-操作說明)
    - [功能模組詳解](#功能模組詳解)
- [實驗數據 (Experimental Results)](#-實驗數據-experimental-results)
- [參數說明 (Configuration)](#-參數說明-configuration)
- [常見問題 (Troubleshooting)](#-常見問題-troubleshooting)

---

## 📝 專案簡介 (Introduction)
在夜間監控、自動駕駛或低光攝影中，傳統影像增強方法（如 HE、Gamma Correction）往往面臨兩難：**提亮了背景，卻讓路燈過曝；看清了輪廓，卻放大了雜訊。**

本專案提出了一種 **「全自適應雙模態融合框架 (Fully Adaptive Dual-Modality Fusion Framework)」**。我們將影像解耦為「光照」與「結構」兩個分量，針對不同區域的統計特徵（亮度、紋理標準差）動態調整增強策略，最終實現 **類 HDR (High Dynamic Range)** 的視覺效果，並顯著提升 YOLO 物件偵測的準確度。

---

## 🌟 核心亮點 (Key Features)

* **🧠 全自適應邏輯 (Fully Adaptive Logic)**
    * 不依賴固定參數，系統自動分析場景的平均亮度 (Mean) 與標準差 (Std Dev)，動態決定補光強度與降噪等級。
* **⚖️ 區域感知融合 (Region-Aware Fusion)**
    * 自動分離 **高光區 (Highlight)** 與 **暗部 (Shadow)**，防止路燈過曝並最大化暗部細節。
* **✨ 智慧去霧與色彩還原 (De-hazing & Color Boost)**
    * 引入 Gamma 壓制與 S 通道增益，消除夜視增強常見的「灰霧感」，還原通透色彩。
* **🤖 下游任務優化 (Machine Vision Ready)**
    * 經 YOLOv8 驗證，增強後的影像能顯著提升暗處物件的 Recall (召回率)。
* **🎛️ 人機協作 GUI (Human-in-the-Loop)**
    * 提供互動式介面，允許使用者在演算法基礎上進行主觀微調。

---

## 🏗️ 系統架構與演算法 (System Architecture)

本系統採用 **雙模態並行處理** 架構：

```mermaid
graph LR
    Input("Input Image") --> HSV["RGB to HSV"]
    HSV --> V["Extract V Channel"]
    
    %% Modality 1
    subgraph Modality_1 [Adaptive Illumination]
    V --> Gamma["Auto-Gamma Injection"]
    Gamma --> Split{"Highlight Detection"}
    Split -->|Mask=1| PathA["Protect (Clip=1.0)"]
    Split -->|Mask=0| PathB["Boost (Clip=8.0)"]
    PathA --> RegionFusion["Region-Aware Fusion"]
    PathB --> RegionFusion
    end
    
    %% Modality 2
    subgraph Modality_2 [Adaptive Structure]
    V -.-> Risk{"Noise Risk Analysis"}
    Risk -->|High Noise| StrongD["Strong Denoise (Sigma=50)"]
    Risk -->|Low Noise| WeakD["Weak Denoise (Sigma=15)"]
    StrongD --> Sobel["Sobel Edge Extraction"]
    WeakD --> Sobel
    end
    
    %% Fusion
    RegionFusion --> Texture{"Texture Analysis"}
    Sobel --> Texture
    Texture -->|High Texture| W_High["Weight=0.4"]
    Texture -->|Smooth Area| W_Zero["Weight=0.0"]
    W_High --> FinalSum["Weighted Sum"]
    W_Zero --> FinalSum
    FinalSum --> Post["De-hazing & Saturation"]
