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

1. 光照模態 (Illumination Modality)檔案： illumination.py原理： 解決動態範圍問題。技術細節：Auto-Gamma: 若平均亮度 < 80，自動注入 Gamma=1.5~3.0 的非線性增益。Masking: 利用閾值 (Threshold > 220) 製作羽化遮罩，將路燈與背景分開處理。CLAHE: 背景區使用極限參數 (Clip=6.0~8.0)，光源區使用保護參數 (Clip=1.0)。2. 結構模態 (Structure Modality)檔案： structure.py原理： 解決細節丟失與雜訊放大問題。技術細節：Dynamic Denoising: 根據區域亮度切換 雙邊濾波 (Bilateral Filter) 強度。暗部使用 Sigma=50 抹平噪點，亮部使用 Sigma=15 保留細節。Gradient Extraction: 使用 Sobel 算子提取純淨的紋理圖 (Structure Map)。3. 自適應融合 (Adaptive Fusion)檔案： fusion.py原理： 重建最終影像質感。技術細節：Zero-Weight Policy: 在平滑區域 (如天空)，強制結構權重為 0，杜絕顆粒感。Gamma De-hazing: 使用 Gamma=1.25 壓制暗部灰階，消除霧霾感。Saturation Boost: 補償 S 通道 (x1.4)，防止色彩泛白。💻 安裝與環境設定 (Installation)1. 系統需求Python 3.8 或以上Windows / Linux / macOS2. 安裝依賴套件建議使用虛擬環境 (Virtual Environment)：Bash# 建立虛擬環境
python -m venv venv
# 啟動虛擬環境 (Windows)
venv\Scripts\activate
# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate
安裝必要函式庫：Bashpip install opencv-python numpy matplotlib scikit-image ultralytics pillow
🚀 使用指南 (Usage Guide)啟動主程式Bashpython main.py
程式啟動後將顯示主選單 (App Launcher)，包含以下功能：GUI 操作說明點擊 "1. Interactive GUI Adjustment" 進入互動模式：Contrast (對比度): 控制 CLAHE 的強度。往右拉，暗部越亮。Detail (細節): 控制結構圖的疊加權重。往右拉，紋理越銳利。Color (色彩): 控制飽和度增益。往右拉，色彩越鮮豔。Denoise (降噪): 控制平滑程度。往右拉，畫面越乾淨（但可能丟失細節）。熱鍵: 按 s 存檔，按 q 退出。功能模組詳解Single Image Enhancement (Adaptive):全自動模式。系統會自動分析圖片並輸出 Modality A、Modality B 與 Final Output 的分解圖。Video Enhancement:支援 .mp4, .avi 格式，將每一幀進行即時增強並存檔。Quantitative Evaluation:需準備 normal_light.jpg 作為 Ground Truth。自動計算 PSNR 與 SSIM，並與 Global HE 進行勝負比較。YOLO Object Detection:自動載入 yolov8n.pt 或 yolov8m.pt。顯示原始圖片與增強圖片的偵測框對比 (Recall 驗證)。Adaptive vs Fixed Experiment:一鍵生成對比圖，展示「自適應算法」如何解決傳統方法的「過曝」與「噪點」問題。📊 實驗數據 (Experimental Results)1. 定量評估 (Quantitative Metrics)MethodPSNR (dB) ↑SSIM ↑備註Baseline (Global HE)14.230.6120雜訊多，亮度不均Ours (Adaptive)19.850.8450結構完整，亮度自然2. 視覺比較 (Visual Comparison)Case 1: 路燈場景 (Highlight Protection)Baseline: 路燈嚴重過曝，變成死白光球。Ours: 燈罩紋理清晰可見，背景依然明亮。Case 2: 極暗泰迪熊 (Extreme Low-Light)Baseline: 勉強可見，但滿是紅綠噪點。Ours: 色彩還原正確，且成功被 YOLO 偵測。⚙️ 參數說明 (Configuration)若需在程式碼層級進行微調，請參考以下關鍵變數：illumination.py:clip_limit: 預設自動判斷 (1.0~8.0)。手動可調範圍通常為 2.0~4.0。mask_threshold: 預設 220 (0-255)。判定是否為高光的閥值。structure.py:sigma: 預設自動判斷 (15 或 50)。雙邊濾波的空間與色彩標準差。fusion.py:gamma_value: 預設 1.25。去霧強度，數值越大黑色越深邃。std_threshold: 預設 45。判定紋理豐富度的閥值。
