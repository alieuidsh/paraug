# paraug

![paraug banner — CPU and GPU augmentation pipelines converging to a single bit-exact output](docs/banner.png)

> **影像增強的 CPU/GPU 位元級精確對齊（bit-exact parity）。**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)]()

**Languages**: [English](README.md) | 繁體中文

`paraug` 是一個通用 PyTorch 原生影像增強函式庫。**35 個 primitive**
（7 geometric + 28 photometric, 含 CutMix / MixUp / GridMask /
RandomErasing）、**GPU batch native**、**CPU/GPU 位元級精確對齊**:
相同的 seed 在 CPU 跟 CUDA 上產生相同的輸出。每個
primitive 的隨機數都在 CPU 端取樣，不受 tensor 在哪個 device 影響 ——
即使訓練流程在 CPU/GPU 階段間隨機切換、或 unit test 換 backend 跑，結果
仍然 deterministic。

需要跨硬體可重現性時可以當 `kornia.augmentation` / `torchvision.v2` 的
drop-in；需要 GPU 加速時可以當 `albumentations` 的 batch-native 替代。

## 為什麼需要 parity

大多數的增強函式庫（albumentations、kornia、torchvision）使用 device-local
RNG。同 seed、CPU/CUDA 結果不同。這在三個場景會踩雷：

1. **可重現性**：論文跟 code 的對應斷裂 —— reviewer 跑不出你 published 的數字。
2. **除錯**：CPU 端的 unit test 抓不到 GPU-only bug，反之亦然。
3. **分散式訓練**：不同硬體的 worker 會 drift 開來。

paraug 的解法是把 RNG 隔離到 CPU（`torch.Generator(device="cpu")`），只把
deterministic 的 torch 運算 route 到 device。容差：

- **Elementwise 運算**（gamma、noise、color jitter…）：atol 1e-6
- **`grid_sample` 類運算**（affine、perspective、tps…）：atol 2e-4
  （ATen 與 cuDNN bilinear 的 ulp drift）

## 安裝

```bash
pip install paraug
```

或從原始碼安裝：

```bash
pip install git+https://github.com/alieuidsh/paraug.git
```

## 快速上手

```python
import torch
from paraug import AugPipeline

# 從 31 個 primitive 裡自己組 config。每個 op 的 `p` 獨立判斷
# (每個 sample 各自決定要不要套這個 op)。
aug = AugPipeline({
    "geometric": {
        "affine": {"p": 1.0, "rot_deg": 15.0, "scale_range": (0.9, 1.1)},
        "tps":    {"p": 0.5, "max_disp": 12.0, "n_ctrl": 5},
    },
    "photometric": {
        "gamma":         {"p": 0.5},
        "color_jitter":  {"p": 0.5},
        "gaussian_blur": {"p": 0.3},
    },
})

# 輸入規格: float tensor in [0, 1], shape (B, C, H, W)。
# (numpy HWC uint8 也接受 — paraug 內部會 normalise)
img  = torch.rand(2, 3, 256, 256)         # (B, C, H, W) in [0, 1]
mask = torch.ones(2, 1, 256, 256)         # 可選的 segmentation mask

# aug 一律回 (img, mask) tuple — 沒 mask 用 `_` 接住:
img_out, mask_out = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
img_only, _      = aug(img,             seed_base=42, epoch=0, step=0)
```

同樣的 call 跑在 GPU 上，輸出在容差內位元級對齊：

```python
img_cuda, mask_cuda = aug(img.cuda(), mask=mask.cuda(),
                           seed_base=42, epoch=0, step=0)
assert (img_out - img_cuda.cpu()).abs().max() < 2e-4
```

### `seed_base`、`epoch`、`step`

三個整數合起來決定 per-item RNG seed (再配合 item 在 batch 內的位置)。
同一個 triple → 同一個 item 同一個輸出。

- **`seed_base`** — run-level seed。pin 在 config 裡，整個訓練 run 都用同一個。
- **`epoch`** — 跨 epoch 換, 讓同一個資料 sample 每個 epoch 都拿到不同 aug。
- **`step`** — epoch 內換, 通常就是 `global_step`。

Inference / 一次性使用時三個都傳 0 就好 (`aug(img, seed_base=0)`)。
拆三個是讓訓練時的 aug 可以 reproducible **而且** 在對的軸上變化 —
不一定要全用。

## paraug 應該放在訓練程式碼的哪裡

第一直覺 (從 `torchvision.transforms` 來的) 是把 aug 塞進
`Dataset.__getitem__`, 讓每個 worker 一個 sample 處理。**paraug 不要這樣** —
它是 batch-native GPU library, 放 per-sample CPU 就把 GPU 加速丟掉了。

```python
# ❌ DON'T — Dataset 內 per-sample CPU aug
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.aug = AugPipeline(cfg)
    def __getitem__(self, idx):
        img = load_image(idx)                    # (C, H, W), CPU
        img, _ = self.aug(img.unsqueeze(0), seed_base=idx)   # CPU aug
        return img.squeeze(0)

# ✅ DO — Dataset 只負責 load, train loop 對 GPU batch 做 aug
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return load_image(idx)                   # 只做 I/O + resize

aug = AugPipeline(cfg, canvas_size=(224, 224))
for step, (images, labels) in enumerate(loader):
    images = images.to(device, non_blocking=True)
    images, _ = aug(images, seed_base=42, epoch=epoch, step=step)
    logits = model(images)
    ...
```

5060 Ti 實測 bs=32 canvas=224×224 (5 個 op 的設定):

| 放法 | wall time / batch | throughput |
|---|---:|---:|
| Dataset 內 per-sample CPU | 219 ms | 146 samples/s |
| **Train loop GPU batch** | **75 ms** | **429 samples/s** |

→ 只是換放的位置就 **2.9× 加速**。batch 越大、canvas 越大、op 越多, 差距越大
(paraug 的 per-op launch overhead 在 batch 內被均攤掉)。

更大 batch / canvas 的設定還可以加
[`set_fast_noise(True)`](#效能調整fast_noise-跟-chunk_size) 跟
[`chunk_size`](#效能調整fast_noise-跟-chunk_size), 見下。

## 合成：`compose(foreground, background, mask)`

`compose` 透過 mask 把 foreground 貼到 background 上，再套設定好的 aug：

```python
from paraug import AugPipeline

aug = AugPipeline({
    "geometric":   {"affine": {"p": 1.0, "rot_deg": 10.0}},
    "photometric": {"gamma": {"p": 0.5}},
})

# numpy (H, W, 3) uint8 進 → numpy 出（也接受 torch tensor）
img, mask = aug.compose(
    foreground = paper_image,   # 要貼上去的紙
    background = scene_image,   # 靜態背景
    mask       = paper_mask,    # 255 = foreground、0 = background
)
```

資料流程：

1. **geometric** primitive 把 `(foreground, mask)` 一起 warp — 前景紙張
   旋轉 / 縮放 / 扭曲，背景維持不動。
2. **blend** — `composite = fg_w * mask_w + background * (1 - mask_w)`。
3. **photometric** primitive 擾動 composite。
4. 選用的 **`canvas_size`** stretch（見下）。

**多層合成**就是呼叫兩次 `compose` — 第一次的輸出當第二次的 foreground：

```python
# 「內容印在紙上 → 紙在場景中被拍照」
img1, m1 = aug.compose(content, paper_tone, content_mask)   # 列印
img2, m2 = aug.compose(img1,    scene_bg,   paper_mask)      # 拍照
```

兩個 pass 要套不同 aug 的話，用兩個 `AugPipeline` 實例。

## 固定輸出尺寸：`canvas_size`

```python
aug = AugPipeline(config, canvas_size=(512, 512))
```

每個 `__call__` / `compose` 輸出都會用非等比 `F.interpolate` 拉成
`(512, 512)` — **不保留**輸入長寬比。當下游 batch 需要統一尺寸、且任務在
stretch 下保持一致時（train 跟 inference 都拉到同一個 canvas，model 在
canvas 空間裡學）這是對的選擇。預設 `None` 讓輸出尺寸等於輸入。

存在 tensor **內**的 GT（`mask`，或用 `n_image_channels` 疊上去的 channel）
會跟著 image 一起 stretch，免費。若 GT 是存 tensor **外**的座標，`compose`
傳 `return_transform=True`，用回傳的 `scale_x` / `scale_y` 換算：

```python
img, mask, t = aug.compose(fg, bg, m, return_transform=True)
line_x = [x * t["scale_x"] for x in line_x]
line_y = [y * t["scale_y"] for y in line_y]
```

## 巢狀矩形 layout：`place_into_canvas`

分割目標如果是**更大畫面中的子區域** —— 而且外圈本身又是矩形 —— 模型容易
偷懶用外圈矩形當答案。訓練時加隨機 layout 才能逼它學「外圈是干擾物要忽略」。

`place_into_canvas` 把 foreground (跟 mask) 放在更大畫布的隨機位置, 周圍填
constant 顏色, 邊距隨機:

```python
from paraug import place_into_canvas

# content: (H, W, 3) uint8 — 真正要分割的內圈區域
# content_mask: (H, W) uint8 — segmentation 目標
padded, padded_mask = place_into_canvas(
    content, content_mask,
    canvas_size=(800, 1000),
    fill=(245, 245, 245),               # 背景填色
    margin_frac_range=(0.05, 0.30),     # 每邊 5-30% margin, 隨機
    seed_base=epoch_step_seed,
)
```

per-item RNG 跟 primitive 同一套 (`seed_base / epoch / step`), CPU/CUDA
bit-exact 重現。

## 效能調整：`fast_noise` 跟 `chunk_size`

兩個 opt-in flag, 在 GPU 上用小小的契約換大的速度 / VRAM 收益。預設都 OFF,
不設的話行為跟上面 doc 一致。

### `paraug.set_fast_noise(True)` — 速度

把三個 CPU-sample noise primitive (`gaussian_noise`, `jpeg_approx`,
`salt_pepper_noise`) 切到 GPU-side `torch.randn` / `torch.rand`。原本
CPU-path 在 canvas 大時是 per-call 時間大戶 (canvas 1024 bs=20 ~350 ms),
因為要在 Python per-item loop 填好再 copy 到 GPU; GPU path ~6 ms。
**5060 Ti bs=20 canvas=1024 + 14 op pipeline 實測 1.85× end-to-end 加速**。

契約: cuRAND ≠ MT19937, 所以 `fast_noise=True` 跟 `fast_noise=False` 同 seed
不會等值。Per `(seed_base, epoch, step)` 的 determinism 在各 mode 內都保留。
跑 parity test 時關掉; production training 開。

### `AugPipeline(cfg, ..., chunk_size=N)` — VRAM

把 batch 內部切成大小 N 的 sub-batches 各跑完整 pipeline 然後 concat。
per-call peak alloc 跟 N 走不跟 batch size 走。**bs=20 → chunk_size=5 實測
peak alloc 降 30-40%**, wall-clock 不變 (chunk 間 cache hit 抵 per-chunk
launch overhead)。

契約: chunked 輸出 per `(seed_base, epoch, step, chunk_size)` deterministic,
但 per-item seed namespace 跟 chunk_size 連動, 改 chunk_size 中間不要期待
bit-equality。

## Optional presets

`paraug.presets` 附幾個調好的 config 給常見部署場景。每個 preset 回 deep-copy
過的 dict 可以再改:

```python
from paraug import AugPipeline, presets
cfg = presets.OOD_PRINTED_PAPER()      # 目前一個, 之後可能加更多
aug = AugPipeline(cfg, canvas_size=(512, 512))
```

Preset **不是**主要 API — 任何 preset 不完全合的任務都建議從 31 個 primitive
自己組 config (見快速上手)。`paraug/presets.py` 看每個 preset 的內容,
`examples/05_ood_printed_paper.py` 是完整的 layered-synthesis 範例。

## 把 GT 當 image 額外 channel 一起 warp

`n_image_channels=N` 宣告：前 N 個 channel 視為「image」（geometric +
photometric 都套），其餘 channel 只走 geometric warp。Photometric primitive
會 skip 額外 channel，所以堆疊在 image 上的 ground-truth field 數值不會被
gamma / noise / shadow 動到，卻又共用同一張 back-warp grid：

```python
import torch
from paraug import AugPipeline

# (B, 3, H, W) RGB + (B, 2, H, W) 整張影像 heatmap GT = 5 channel
img_rgb  = torch.rand(2, 3, 256, 256)
gt_h     = render_h_line_heatmap(...)   # 自己的 renderer; (B, 1, H, W)
gt_v     = render_v_line_heatmap(...)   # (B, 1, H, W)
img_5ch  = torch.cat([img_rgb, gt_h, gt_v], dim=1)   # (B, 5, H, W)

aug = AugPipeline({
    "geometric":   {"affine": {"p": 1.0, "rot_deg": 10.0},
                      "tps":    {"p": 0.5, "max_disp": 8.0, "n_ctrl": 5}},
    "photometric": {"gamma": {"p": 0.5, "gamma_range": (0.8, 1.2)}},
}, n_image_channels=3)

out, _ = aug(img_5ch, seed_base=42)
# out[:, :3] = warp + gamma 後的 RGB
# out[:, 3:] = 只 warp 沒 gamma 的 heatmap
```

GT 是 2-D field 的任務（line heatmap、連續標籤分割、distance transform、
切線場）原本要解 forward-warp 才能跟著 image 對齊，現在直接把 GT 當 channel
堆上去 + 同一次 `grid_sample`，省下整套 forward-warp solver。預設
`n_image_channels=None` 保留原本行為，沒設就不切。

`random_shadow` 雖然 dispatch 屬 geometric，但效果是乘法的 photometric，
split 啟動時會被當 photometric 處理（不會把陰影因子套在 GT channel 上）。

### Sampling mode 說明（`mask` vs 額外 channel）

堆在 `img` 後面的額外 channel **跟 image 一樣 bilinear interpolation**。如果
你的 GT 是整數 class label / segmentation ID 不能線性插值，要走
**nearest**，要把那個 tensor 用 `mask=` 參數傳，不要塞進 `img`：

| 路徑 | Interpolation | 套 photometric 嗎 | Channel 數 |
|------|--------------|------------------|-----------|
| `img[:, :n_image_channels]` (RGB / image) | bilinear | 套 | 任意 |
| `img[:, n_image_channels:]` (額外) | bilinear | **不套** | 任意 |
| `mask` 參數 | **nearest** | 不套 | 1 (單通道) |

paraug 的 geometric primitive 對 `img` 跟 `mask` 用同一個 back-warp grid，
只差在 interpolation mode。Photometric primitive 永遠不動 `mask`。

## Primitive 列表

### Geometric（7 個）

| 名稱 | 說明 |
|---|---|
| `affine` | 透過 `F.affine_grid` 做旋轉 + 縮放 + 平移 |
| `perspective` | 四角點 jitter 解出 homography |
| `random_crop_pad` | 縮放後 pad 回原尺寸 |
| `elastic_transform` | 低解析度隨機 displacement field bilinear 上取樣 |
| `optical_distortion` | 徑向 barrel / pincushion 失真（k·r²）|
| `random_shadow` | 軟邊三角形乘法陰影 |
| `tps` | 低解析度控制網格的 thin-plate-spline 類 warp |

### Photometric（24 個）

亮度 / 顏色：`gamma`、`color_jitter`、`hue_shift`、`random_grayscale`、
`lighting`、`clahe`、`local_contrast`、`sharpness`。

雜訊：`gaussian_noise`、`salt_pepper_noise`、`salt_patches`。

模糊 / 壓縮：`gaussian_blur`、`motion_blur`、`jpeg_approx`。

光線：`vignette`、`specular_highlight`、`specular_streaks`。

內容疊加：`cutout`、`paper_texture_overlay`、`watermark`、
`random_text_overlay`、`background_compose`、`stains`、`creases`。

## Parity 對比

| 函式庫 | Bit-exact CPU↔GPU | Per-item RNG | GPU 原生 | Mask-aware | Batch-native | # Geometric¹ | # Photometric¹ | License |
|---|---|---|---|---|---|---|---|---|
| **paraug** | **✓**（1e-6 / 2e-4）² | ✓ | ✓ (torch) | ✓ | ✓ | 7 | 24 | Apache 2.0 |
| albumentations | ✗（numpy-only）| ✓ | ✗ | ✓ | 部分 | ~20 | ~50+ | MIT |
| kornia | ✗（device-local RNG）| ✓ | ✓ (torch) | ✓ | ✓ | ~10 | ~45 | Apache 2.0 |
| torchvision.v2 | ✗（device-local RNG）| ✓ | ✓ (torch) | 部分 | ✓ | ~18 | ~12 | BSD-3 |
| imgaug | ✗（numpy-only）| ✓ | ✗ | ✓ | 部分 | ~20 | ~40 | MIT |
| augly | ✗（PIL-only）| ✓ | ✗ | ✗ | ✗ | ~5 | ~20 | MIT |

<sub>¹ 其他函式庫的計數為 2026-05 sample 自各 project 的 `__init__.py` /
docs index，數字會隨版本變動 —— 請以各 project 的官方 API 為準。paraug
的計數來自 code（`len(GEOMETRIC_PRIMITIVES)` / `len(PHOTOMETRIC_PRIMITIVES)`），
精確值。</sub>

<sub>² 容差由 `tests/test_parity.py` 在 NVIDIA 5060 Ti + 4080 上於 v0.1.0
驗證：1e-6 適用於 `PHOTO_ELEMENTWISE` 列出的 6 個純 elementwise photometric
運算（gamma / gaussian_noise / color_jitter / vignette / cutout /
hue_shift），2e-4 適用於 grid_sample 類（geometric）跟 conv 類（blur）運算。
**GitHub 免費 CI runner 沒 GPU**，所以 13 個 CUDA parity test 在 CI 上 skip
—— 歡迎社群在其他 GPU SKU 上驗證後開 PR 補上結果，或本機跑 `pytest
tests/test_parity.py -k cpu_vs_cuda` 把輸出貼上來。</sub>

### 適合用 paraug 的場景

- 跨 device 的可重現性（paper-grade ablation 不能讓 CPU↔GPU 漂移破壞 baseline）
- 異質硬體的分散式訓練
- 對 unit test 友善的增強流程（CPU-side RNG → 免費 CI runner 就能重現開發者 GPU 上的結果）

### **不**適合用 paraug 的場景

- 你需要 50+ 種 primitive 開箱即用 → 試 `albumentations` 或 `imgaug`
- 你需要 PIL 風格的 image-by-image API → 試 `augly`
- 你需要內建的組合 op 像 `OneOf` / `SomeOf` → 試 `albumentations`

## 範例

見 `examples/`：

- `01_quickstart.py` — 最小的 load → augment → save 流程
- `02_mask_aware.py` — image + segmentation mask 一起 warp
- `03_cpu_gpu_parity.py` — 同 seed 跑 CPU 與 CUDA，assert
  `max_abs_diff < 2e-4`

## 引用

```bibtex
@software{paraug2026,
  author = {alieuidsh},
  title  = {paraug: Bit-exact CPU/GPU parity for image augmentation},
  year   = {2026},
  url    = {https://github.com/alieuidsh/paraug},
}
```

## License

Apache 2.0 —— 詳見 [LICENSE](LICENSE)。
