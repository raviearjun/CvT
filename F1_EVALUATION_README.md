# F1 Score Evaluation Implementation

## 📋 Overview

Implementasi evaluasi telah diperbarui untuk menggunakan **F1 Score** sebagai metrik utama dalam proses training, sambil tetap mempertahankan **Accuracy** untuk perbandingan.

## 🔧 Fitur Baru

### 1. **F1 Score sebagai Metrik Primer**

- F1 Score digunakan sebagai metrik utama untuk pemilihan model terbaik
- Lebih cocok untuk dataset yang tidak seimbang (imbalanced)
- Support untuk berbagai averaging methods: `macro`, `weighted`, `micro`

### 2. **Konfigurasi Fleksibel**

```yaml
TRAIN:
  USE_F1_AS_PRIMARY: true # Gunakan F1 sebagai metrik utama
  F1_AVERAGE: "macro" # Method averaging: macro/weighted/micro
```

### 3. **Dual Metrics Logging**

- F1 Score: Metrik utama
- Accuracy: Metrik sekunder untuk perbandingan
- Keduanya dicatat di tensorboard dan log training

## 📊 Output Contoh

### Training Log:

```
=> Epoch[1][50/100]: Time 2.345s (2.123s) Speed 15.2 samples/s Data 0.012s (0.011s)
   Loss 0.85432 (0.92156) F1-Score 85.6 (83.2) Accuracy@1 87.3 (85.1) Accuracy@5 96.2 (94.8)
```

### Validation Log:

```
=> TEST: Loss 0.7234 F1-Score 89.45% Error@1 12.3% Error@5 3.2% Accuracy@1 87.7% Accuracy@5 96.8%
=> Current F1-Score: 89.45%, Accuracy: 87.70%
=> Best F1-Score so far: 89.45%
```

## 🔄 Backward Compatibility

Kode lama tetap berfungsi:

- Fungsi `accuracy()` masih ada dan tidak berubah
- Model selection sekarang berdasarkan F1, bukan accuracy
- Semua log dan tensorboard tetap mencatat kedua metrik

## ⚙️ Konfigurasi F1 Average Methods

### **Macro Average** (Default - Recommended)

```yaml
TRAIN:
  F1_AVERAGE: "macro"
```

- Hitung F1 untuk setiap kelas, lalu rata-rata
- Ideal untuk menilai performa di semua kelas secara setara
- **Direkomendasikan untuk dataset imbalanced**

### **Weighted Average**

```yaml
TRAIN:
  F1_AVERAGE: "weighted"
```

- F1 untuk setiap kelas diweight berdasarkan jumlah sampel
- Memberikan penalti lebih pada kelas dengan sampel banyak

### **Micro Average**

```yaml
TRAIN:
  F1_AVERAGE: "micro"
```

- Hitung TP, FP, FN secara global di semua kelas
- Biasanya sama dengan accuracy pada multiclass

## 🚀 Penggunaan

### 1. **Update Config YAML**

Tambahkan ke file konfigurasi Anda:

```yaml
TRAIN:
  USE_F1_AS_PRIMARY: true
  F1_AVERAGE: "macro" # Sesuaikan dengan kebutuhan
```

### 2. **Jalankan Training**

```bash
python tools/train.py --config experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml
```

### 3. **Monitor Metrics**

- **Tensorboard**: `train_f1`, `valid_f1`, `train_top1`, `valid_top1`
- **Logs**: F1-Score dan Accuracy dicatat bersamaan
- **Model Selection**: Best model dipilih berdasarkan F1 tertinggi

## ✅ Keuntungan F1 Score

1. **Better for Imbalanced Datasets**: Lebih sensitif terhadap class imbalance
2. **Harmonic Mean**: Menyeimbangkan precision dan recall
3. **Per-class Evaluation**: Memberikan insight lebih baik pada performa per kelas
4. **Standard Metric**: Umum digunakan dalam kompetisi ML dan research

## 🔍 Files yang Dimodifikasi

1. **`lib/core/evaluate.py`**: Menambahkan fungsi F1 score
2. **`lib/core/function.py`**: Mengintegrasikan F1 ke training loop
3. **`lib/config/default.py`**: Konfigurasi F1 parameters
4. **`tools/train.py`**: Model selection berdasarkan F1

## 📈 Interpretasi Hasil

- **F1 Score > 85%**: Performa sangat baik
- **F1 Score 70-85%**: Performa baik
- **F1 Score 50-70%**: Performa sedang
- **F1 Score < 50%**: Performa kurang baik

**Catatan**: F1 Score umumnya lebih konservatif dari Accuracy, jadi nilai yang sedikit lebih rendah adalah normal.

## 🎯 Best Practices

1. **Gunakan `macro` average** untuk dataset dengan class imbalance
2. **Monitor kedua metrik** (F1 dan Accuracy) untuk insight lengkap
3. **Validasi dengan confusion matrix** untuk analisis per-kelas yang detail
4. **Adjust F1 averaging method** sesuai dengan karakteristik dataset Anda

---

## 📝 Example Configuration

```yaml
OUTPUT_DIR: "/kaggle/working/"
WORKERS: 4
PRINT_FREQ: 50

MODEL:
  NAME: cls_cvt
  NUM_CLASSES: 5

TRAIN:
  BATCH_SIZE_PER_GPU: 8
  LR: 0.0005
  END_EPOCH: 100

  # F1 Score Configuration
  USE_F1_AS_PRIMARY: true
  F1_AVERAGE: "macro" # Recommended for cultural classification
  EVAL_BEGIN_EPOCH: 0

DATASET:
  DATASET: "imagenet"
  ROOT: "/kaggle/working/CvT/cultural_dataset/"
  SAMPLER: weighted # Complement F1 with weighted sampling
```

Dengan konfigurasi ini, model akan dioptimalkan menggunakan F1 Score yang lebih cocok untuk task klasifikasi budaya dengan 5 kelas (balinese, batak, dayak, javanese, minangkabau).
