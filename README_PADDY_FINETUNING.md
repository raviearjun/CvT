# CvT Fine-tuning untuk Klasifikasi Penyakit Tanaman Padi

Panduan lengkap untuk melakukan fine-tuning model CvT-21 untuk klasifikasi penyakit tanaman padi dengan 10 kelas.

## Setup di Google Colab

### 1. Persiapan Dataset

Struktur dataset harus dalam format ImageFolder dan diletakkan di root repo:

```
CvT/
├── paddy_disease_classification/
│   ├── train/
│   │   ├── bacterial_leaf_blight/
│   │   ├── bacterial_leaf_streak/
│   │   ├── bacterial_panicle_blight/
│   │   ├── blast/
│   │   ├── brown_spot/
│   │   ├── dead_heart/
│   │   ├── downy_mildew/
│   │   ├── hispa/
│   │   ├── normal/
│   │   └── tungro/
│   └── val/
│       ├── bacterial_leaf_blight/
│       ├── bacterial_leaf_streak/
│       ├── bacterial_panicle_blight/
│       ├── blast/
│       ├── brown_spot/
│       ├── dead_heart/
│       ├── downy_mildew/
│       ├── hispa/
│       ├── normal/
│       └── tungro/
└── CvT-21-224x224-IN-1k.pth
```

### 2. Persiapan File

Upload ke Google Drive:

- Dataset dengan nama folder `paddy_disease_classification` dalam struktur di atas
- File pretrained weights: `CvT-21-224x224-IN-1k.pth`

### 3. Jalankan Training

1. Buka `run_training_colab.ipynb` di Google Colab
2. Jalankan semua cell secara berurutan
3. Sesuaikan path dataset dan weights di cell ke-3

## Konfigurasi Training

File konfigurasi: `experiments/imagenet/cvt/cvt-21-224x224.yaml` (sudah diedit untuk fine-tuning)

### Parameter Utama:

- **Jumlah kelas**: 10 (9 penyakit + 1 normal)
- **Batch size**: 32
- **Image size**: 224x224
- **Epochs**: 30
- **Learning rate**: 5e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing

### Augmentasi Data:

- Random resize crop
- Color jitter
- Horizontal flip
- Normalization (ImageNet stats)

## Monitoring Training

### 1. Real-time Monitoring

```python
# Di cell terpisah, jalankan untuk melihat progress
!tail -f /content/output/log.txt
```

### 2. TensorBoard (opsional)

```python
%load_ext tensorboard
%tensorboard --logdir /content/output
```

## Output Training

Setelah training selesai, file berikut akan tersedia di `/content/output/`:

- `best.pth`: Model dengan validation accuracy terbaik
- `latest.pth`: Model dari epoch terakhir
- `log.txt`: Log lengkap training
- `events.out.tfevents.*`: File TensorBoard

## Evaluasi Model

### 1. Evaluasi pada Test Set

```bash
python tools/test.py \
    --cfg experiments/imagenet/cvt/cvt-21-224x224.yaml \
    --model-file /content/output/best.pth
```

### 2. Inference pada Gambar Baru

Untuk inference pada gambar tunggal, Anda perlu memodifikasi `tools/test.py` atau membuat script terpisah.

## Tips Optimisasi

### 1. Jika Memory Error:

- Kurangi batch size dari 32 ke 16 atau 8
- Edit `TRAIN.BATCH_SIZE_PER_GPU` di file `experiments/imagenet/cvt/cvt-21-224x224.yaml`

### 2. Jika Training Lambat:

- Gunakan GPU T4 atau lebih tinggi di Colab Pro
- Kurangi resolusi image dari 224 ke 192

### 3. Jika Overfitting:

- Tingkatkan `LOSS.LABEL_SMOOTHING` dari 0.1 ke 0.2
- Tambahkan dropout dengan mengedit `MODEL.SPEC.DROP_RATE`

## Troubleshooting

### Error: "CUDA out of memory"

```yaml
# Di experiments/imagenet/cvt/cvt-21-224x224.yaml, ubah:
TRAIN:
  BATCH_SIZE_PER_GPU: 16 # atau 8
```

### Error: "FileNotFoundError" untuk dataset

Pastikan path dataset di config benar:

```yaml
DATASET:
  ROOT: "/content/CvT/paddy_disease_classification/" # Sesuaikan dengan lokasi dataset
```

### Error: "Model checkpoint not found"

Pastikan file pretrained weights ada:

```yaml
MODEL:
  PRETRAINED: "/content/CvT/CvT-21-224x224-IN-1k.pth"
```

## Backup Hasil

Semua hasil training akan otomatis disimpan ke Google Drive di folder:
`/content/drive/MyDrive/CvT_Paddy_Results/`

## Kustomisasi

### 1. Mengganti Jumlah Kelas

Edit di `experiments/imagenet/cvt/cvt-21-224x224.yaml`:

```yaml
MODEL:
  NUM_CLASSES: 10 # Ubah sesuai jumlah kelas Anda
```

### 2. Mengganti Arsitektur Model

Untuk menggunakan CvT-13 atau CvT-W24:

```yaml
MODEL:
  PRETRAINED: "/path/to/CvT-13-224x224-IN-1k.pth" # atau CvT-W24
  SPEC:
    # Sesuaikan parameter arsitektur
```

### 3. Transfer Learning vs Fine-tuning

Untuk freeze backbone dan hanya train classifier:

```yaml
FINETUNE:
  FROZEN_LAYERS: ["backbone"] # Freeze semua layer kecuali head
```

## Kontribusi

Jika menemukan bug atau ingin berkontribusi, silakan buat issue atau pull request di repository ini.

## Lisensi

Mengikuti lisensi dari repository Microsoft CvT original.
