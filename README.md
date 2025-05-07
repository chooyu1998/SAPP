# SAPP: Structure-Aware PTM Prediction

SAPP is a transformer-based model for post-translational modification (PTM) site prediction, integrating protein sequence and structural features (RSA).

![finetuningarch_new](https://github.com/user-attachments/assets/5d68237d-892d-4256-8aa0-32bd24298e1b)



---

## ✅ Features

- Supports 7 PTM types: phosphorylation (S/Y), methylation (K/R), acetylation (K), ubiquitination (K), SUMOylation (K)
- Supports 8 Kinase-specific types: CMGC, CAMK, CDK, AGC, MAPK, PKA, PKC, CK2
- Accepts RSA input in `.npy`, `.cif`, or `.pdb` formats
- Automatically computes RSA from AF structure files if `.npy` not provided
- Multi-checkpoint ensemble prediction
- GPU acceleration supported

---

## 📁 Input Format

A tab-separated `.txt` file with the following columns:

```text
ProteinID    Sequence    Site    Label    PTMType    RSA_or_AF_path
```

### Example:
```text
P12345    MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYR    34    1    SAPPPhos    data/RSA_files/P12345.npy
Q67890    MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWMAKH    17    0    SAPPmethylK    data/AF_files/Q67890.cif
```

---

## 🚀 Inference Usage

```bash
python inference.py \
  --input example_folder/inference_input_wRSA.txt \
  --output example_folder/result.csv \
  --device cuda:0
```

### Optional Arguments
- `--batch_size`: Default = 128
- `--device`: e.g., `cuda:0` or `cpu`

---

## 📂 Model Checkpoints

Organize model weights per PTM type:

```
SAPP/
└── data/
    └── models/
        ├── SAPPPhos/
        │   ├── 1Fold-crossvalidation.pt
        │   └── 2Fold-crossvalidation.pt
        └── ...
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Output Format

The resulting CSV contains:

| ProteinID | Site | Pred | Label | PTMType |
|-----------|------|------|-------|---------|
| Q12345    | 34   | 0.88 | 1     | SAPPPhos |


