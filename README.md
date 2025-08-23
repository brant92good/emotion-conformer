# Wav2Vec2-Conformer for ASVspoof (heavily modified)

這個專案最初用於情緒辨識（IEMOCAP），現已「大幅修改」以聚焦在 ASVspoof 偵測（目前預設為 ASVspoof 2019 LA，二分類：bonafide vs spoof）。原本與情緒任務相關的腳本與文件仍保留於 repo 中，但已屬於舊版（legacy），以下文件以 ASVspoof 為主。

## 功能概覽

- 前端使用 fairseq Wav2Vec2/Conformer 預訓練模型（可選擇凍結參數）。
- 簡潔的下游分類器（線性層 + ReLU + Dropout + 線性層）進行二分類：bonafide、spoof。
- 內建 ASVspoof 2019 LA 資料載入與訓練/驗證/測試切分（train/dev）。
- 使用 Hydra 設定檔（`config/asv19.yaml`），支援 TensorBoard 紀錄。
- 仍保留 IEMOCAP 的資料前處理/訓練程式作為參考，但非本分支主要目標。

## 專案結構（重點）

- `src/model.py`
    - `ASVspoofWav2vec2Conformer`：ASVspoof 專用模型類別。
- `src/dataset.py`
    - `ASVspoof2019_Dataset`、`ASVspoof2019_DataLoader`：以 cm protocol + 音檔路徑組建資料集與 DataLoader。
    - `PadCollator`：固定長度 padding 與 mask 處理。
- `src/train.py`
    - 預設使用 `config/asv19.yaml` 啟動訓練流程（k-fold 參數存在，但預設 fold=1）。
- `config/asv19.yaml`
    - 設定模型與資料路徑、訓練超參數。
- `script/` 下 Shell 腳本為舊版情緒辨識流程（LibriSpeech/IEMOCAP）示例，ASVspoof 不需使用。

## 環境需求與安裝

參考 `pyproject.toml`：

- Python: 3.10.17
- 依賴（節錄）：
    - `fairseq==0.12.2`
    - `torch==2.7.1`
    - `torchaudio==2.7.1`
    - `soundfile`, `tensorboard`, `nvitop`, `scikit-learn`, `matplotlib`, `seaborn`

注意：`pyproject.toml` 透過 uv 指定了 fairseq 的本地來源 `[tool.uv.sources] fairseq = { path = "../fairseq", editable = true }`。若你的 fairseq 路徑不同（本倉庫中為 `fairseq-a5402.../`），請調整為正確位置（例如改成相對於本專案的實際路徑）或建立對應的 symlink。

安裝（二擇一）：

- 使用 uv
    - 建立虛擬環境並安裝依賴：
        1) 安裝 uv（若尚未安裝）。
        2) 在專案根目錄執行安裝，確保 fairseq 路徑設定正確。

- 使用 pip
    - 建議啟用 virtualenv/conda，並逐一安裝 `pyproject.toml` 中列出的套件與版本。

## 資料準備（ASVspoof 2019 LA）

請先準備 ASVspoof 2019 LA 的音檔與 cm protocol 檔，並在 `config/asv19.yaml` 填入正確路徑：

- `dataset.train_metadata_path`：例如 `ASVspoof2019.LA.cm.train.trn.txt`
- `dataset.train_audio_folder`：對應 train 音檔根資料夾
- `dataset.dev_metadata_path`：例如 `ASVspoof2019.LA.cm.dev.trl.txt`
- `dataset.dev_audio_folder`：對應 dev 音檔根資料夾
- `dataset.audio_extension`：預設 `.flac`（依你的資料實際副檔名調整）

關於標籤：本實作以 `bonafide` 與 `spoof` 兩類為準。cm protocol 的最後一欄通常為 `bonafide` 或 `spoof`，程式會自動對應為 0/1。

## 設定檔重點（`config/asv19.yaml`）

- `model.frontend_model.path`：fairseq 預訓練權重（checkpoint）路徑。
- `model.hidden_dim`：分類器中介維度（如 384）。
- `dataset.*`：資料路徑與批次設定。
- `train.fold`、`train.epoch`：訓練迭代設定。
- `optimizer`、`scheduler`：優化器與學習率排程。

小提醒：
- 目前程式中部分欄位命名有新舊混用痕跡，例如 `ratios` vs `ratio`、`lable_map`（拼字）等，請以範本檔案 `config/asv19.yaml` 為準並保持一致；若自訂設定檔請使用相同鍵名。
- `dataset.audio_extension` 必須與實際檔案一致，否則會找不到音檔。

## 訓練

基本步驟：

1) 編輯 `config/asv19.yaml`，確認資料與 checkpoint 路徑。
2) 執行訓練：

     - 從專案根目錄執行：`python src/train.py`
     - 程式會使用 Hydra 將輸出（log、checkpoint、tensorboard）寫到執行目錄下的 `checkpoint/` 與 `tensorboard/`。

TensorBoard：啟動指令見訓練 log 內提示，預設為 `tensorboard --logdir=./tensorboard --host=0.0.0.0 --port=6006`。

評估指標：
- 目前訓練流程內建計算的是 classification accuracy（val/test）。若要計算 ASVspoof 常見的 EER/t-DCF，可在推論階段呼叫模型的 `predict_proba` 或 `get_bonafide_scores` 取得分數，再以外部工具計算。

## 推論與 EER 計算（範例）

以下展示如何載入訓練好的模型並取得 bonafide 機率分數，之後可自行換算成 EER：

```python
import torch
from src.model import ASVspoofWav2vec2Conformer

# 準備輸入：waveform (B, T) 與 padding_mask (B, T)
waveform = ...        # torch.Tensor, 16kHz 單聲道
padding_mask = ...    # torch.BoolTensor, True 表示 padding 位置

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ASVspoofWav2vec2Conformer(
        checkpoint_path='/path/to/fairseq_checkpoint.pt',
        hidden_dim=384,
        freeze_frontend=True,
        device=DEVICE,
)
state = torch.load('/path/to/your_trained_checkpoint.pt', map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

with torch.no_grad():
        # 取 logits
        logits = model(waveform, padding_mask)
        # 或取 bonafide 機率（[:, 0]）
        p_bonafide = model.get_bonafide_scores(waveform, padding_mask)
```

注意：`src/inference.py` 仍是舊版情緒任務的推論流程，不適用於 ASVspoof，僅供參考。

## 已知議題與差異

- `src/train.py` 在程式開頭有以「絕對路徑」讀取 `config/asv19.yaml` 的片段，且路徑示例為 `/home/brant/Projects/emotion-conformer/...`（注意 `Projects`）——請依你的實際環境調整，或直接改為相對路徑；Hydra 會在主函式內再載入一次設定檔。
- `config/asv19.yaml` 與程式碼間的欄位名稱有個別歷史遺留：
    - `dataset.ratios` vs `dataset.ratio`；請使用目前檔案中的鍵名以避免 KeyError。
    - `dataset.lable_map` 拼字為舊名（僅少數程式片段參考到）。ASV 任務實際上不需要情緒的 label map。
- `script/` 內的 shell 檔多為情緒任務（LibriSpeech/IEMOCAP）示例，含硬編碼路徑，ASVspoof 不需使用。
- `pyproject.toml` 指向 `../fairseq` 的本地來源，請務必修正為你本機實際 fairseq 位置或使用 pip 直接安裝對應版本的 fairseq。

## 授權

本專案沿用上游與相依套件的授權條款。請遵循 ASVspoof 資料集使用規範與 fairseq/PyTorch 的授權。