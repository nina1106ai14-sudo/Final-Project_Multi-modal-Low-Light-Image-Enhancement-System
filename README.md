# Multi-modal Low-Light Image Enhancement System
# 基於多模態融合的全自適應低光影像增強系統

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)

這是一個基於電腦視覺 (Computer Vision) 與影像處理 (DIP) 技術的低光影像增強系統。本專案不依賴大型深度學習網路進行影像生成，而是採用**「雙模態分解與融合 (Dual-Modality Decomposition & Fusion)」**策略，結合**全自適應 (Fully Adaptive)** 演算法，實現高效、可解釋且強健的夜間影像增強。

## 🌟 核心功能 (Key Features)

* **全自適應光照增強 (Adaptive Illumination):** * 採用 **區域感知融合 (Region-Aware Fusion)**，自動分離光源（如路燈）與背景。
    * 針對暗部進行強力補光，同時保護高光區域，避免過曝 (Over-exposure)。
    * 引入 **Auto-Gamma** 機制，針對極暗場景自動注入增益。
* **動態結構提取 (Dynamic Structure Extraction):**
    * 基於亮度風險分析 (Noise Risk Analysis)，動態調整雙邊濾波 (Bilateral Filter) 強度。
    * 有效在保留紋理細節的同時，抑制暗部雜訊放大。
* **智慧融合與去霧 (Smart Fusion & De-hazing):**
    * 根據紋理豐富度 (Std Dev) 動態調整融合權重。
    * 引入 **Gamma De-hazing** 與 **Saturation Boost**，消除夜視增強常見的灰霧感。
* **人機協作 GUI (Human-in-the-Loop):**
    * 提供互動式介面，使用者可即時調整對比、細節、色彩與降噪參數。
* **下游任務驗證 (Downstream Task Validation):**
    * 整合 **YOLOv8** 物件偵測，驗證增強後的影像能顯著提升機器視覺的召回率 (Recall)。

## 🛠️ 系統架構 (System Architecture)

本系統將影像解耦為 **光照模態 (Illumination)** 與 **結構模態 (Structure)** 兩條平行路徑處理：

```mermaid
graph LR
    %% 樣式定義
    classDef input fill:#ffffff,stroke:#000000,stroke-width:2px,font-weight:bold;
    classDef output fill:#ccff90,stroke:#33691e,stroke-width:2px,font-weight:bold;
    classDef process fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef decision fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,stroke-dasharray: 5 5;

    %% 流程圖
    Input("Input Image"):::input --> HSV["RGB to HSV"]:::process
    HSV --> V["Extract V Channel"]:::process
    
    %% 光照路徑
    V --> Gamma["Gamma Correction"]:::process
    Gamma --> CheckLight{"Highlight Check"}:::decision
    CheckLight -->|"High Light"| PathA["Protect (Clip=1.0)"]:::process
    CheckLight -->|"Dark Area"| PathB["Boost (Clip=6.0)"]:::process
    PathA --> RegionFusion["Region-Aware Fusion"]:::process
    PathB --> RegionFusion
    
    %% 結構路徑
    V -.-> CheckNoise{"Noise Risk (Mean)"}:::decision
    CheckNoise -->|"High Noise"| DenoiseA["Strong Denoise"]:::process
    CheckNoise -->|"Low Noise"| DenoiseB["Weak Denoise"]:::process
    DenoiseA --> Sobel["Sobel Operator"]:::process
    DenoiseB --> Sobel
    
    %% 融合
    RegionFusion --> Texture{"Texture Analysis"}:::decision
    Sobel --> Fusion["Weighted Sum"]:::process
    Texture --> Fusion
    Fusion --> Post["De-hazing & Saturation"]:::output
