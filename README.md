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
# 🏗️ 系統架構與演算法 (System Architecture)

本系統採用 **雙模態並行處理** 架構，將影像解耦為不同分量進行針對性增強。

### 1. 光照模態 (Illumination Modality)
* **檔案：** `illumination.py`
* **原理：** 解決動態範圍問題。
* **技術細節：**
    * **Auto-Gamma:** 若平均亮度 `< 80`，自動注入 `Gamma=1.5~3.0` 的非線性增益，提升基底亮度。
    * **Masking:** 利用閾值 (`Threshold > 220`) 製作羽化遮罩，將路燈（高光）與背景分開處理。
    * **CLAHE:** 採用區域感知策略，背景區使用極限參數 (`Clip=6.0~8.0`) 以挖掘細節，光源區使用保護參數 (`Clip=1.0`) 以防止過曝。

### 2. 結構模態 (Structure Modality)
* **檔案：** `structure.py`
* **原理：** 解決細節丟失與雜訊放大問題。
* **技術細節：**
    * **Dynamic Denoising:** 根據區域亮度動態切換 **雙邊濾波 (Bilateral Filter)** 強度。
        * **暗部:** 使用 `Sigma=50` 抹平高風險噪點。
        * **亮部:** 使用 `Sigma=15` 保留細節。
    * **Gradient Extraction:** 使用 Sobel 算子提取純淨的紋理圖 (Structure Map)。

### 3. 自適應融合 (Adaptive Fusion)
* **檔案：** `fusion.py`
* **原理：** 重建最終影像質感。
* **技術細節：**
    * **Zero-Weight Policy:** 在平滑區域（如天空），強制結構權重為 `0`，杜絕顆粒感。
    * **Gamma De-hazing:** 使用 `Gamma=1.25` 壓制暗部灰階，消除提亮後常見的霧霾感。
    * **Saturation Boost:** 對 S 通道進行補償 (`x1.4`)，防止色彩因亮度提升而泛白。

---

## 💻 安裝與環境設定 (Installation)

### 1. 系統需求
* Python 3.8 或以上
* Windows / Linux / macOS

### 2. 安裝依賴套件
建議使用虛擬環境 (Virtual Environment)：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate
