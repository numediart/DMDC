---

# Guide complet d’installation WhisperX avec CUDA 11.8 et cuDNN via Conda sur Ubuntu 24.04

---

## 1. Installer Miniconda (si ce n’est pas déjà fait)

```bash
# Téléchargement
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Installation
bash Miniconda3-latest-Linux-x86_64.sh

# Suivez les instructions, acceptez la licence et choisissez d’ajouter conda au PATH

# Recharge le shell
source ~/.bashrc

# Vérifie que conda est installé
conda --version
```

---

## 2. Créer un environnement conda dédié

```bash
conda create -n whisperx python=3.10 -y
conda activate whisperx
```

---

## 3. Installer CUDA 11.8 et cuDNN compatibles

### Ajouter le canal NVIDIA CUDA 11.8

```bash
conda config --add channels nvidia/label/cuda-11.8.0
conda config --add channels conda-forge
conda config --add channels defaults
```

### Installer CUDA 11.8 + cuDNN 8.6 (adapté à WhisperX)

```bash
conda install -c nvidia cudnn=8
# export LD_LIBRARY_PATH=/home/hugo-mny/miniconda3/envs/whisperx/lib/libcudnn_ops_infer.so.8

```


---

## 4. Installer PyTorch avec CUDA 11.8

```bash
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y
```

---

## 5. Installer WhisperX et dépendances Python

```bash
pip install --upgrade pip
pip install whisperx
# Installe aussi les dépendances si besoin, ex:
pip install numpy scipy
```

---

## 6. Configurer `LD_LIBRARY_PATH` pour cuDNN

### Trouver le chemin exact où cuDNN est installé dans conda

```bash
echo $CONDA_PREFIX
# Exemple: /home/hugo-mny/miniconda3/envs/whisperx
```

```bash
find $CONDA_PREFIX -name "libcudnn_ops_infer.so.8"
```

> Normalement, tu devrais avoir un chemin du genre :
> `/home/hugo-mny/miniconda3/envs/whisperx/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8`

### Exporter la variable d’environnement

```bash
export LD_LIBRARY_PATH=/home/hugo-mny/miniconda3/envs/whisperx/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

```

---

## 7. Rendre `LD_LIBRARY_PATH` permanent

Ajouter dans `~/.bashrc` ou `~/.zshrc` :

```bash
export LD_LIBRARY_PATH=/home/hugo-mny/miniconda3/envs/whisperx/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

Puis :

```bash
source ~/.bashrc
```

---

## 8. Vérifications rapides

* Test PyTorch et cuDNN :

```bash
python -c "import torch; print(torch.backends.cudnn.version())"
```

* Vérifier que la lib cuDNN est trouvée par `ldd` :

```bash
ldd $(python -c "import torch; print(torch.utils.cmake_prefix_path)") | grep cudnn
```

---


### Versions incompatibles cuDNN / CUDA

* Vérifie ta version CUDA (exemple : `nvcc --version` ou `nvidia-smi`)
* Vérifie que ta version de cuDNN est compatible avec CUDA installée (cf. doc NVIDIA)

---
