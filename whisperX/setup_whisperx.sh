
python3 -m venv .whisperx-env
source .whisperx-env/bin/activate
pip install --upgrade pip

# Force install the last torch version with cuDNN 8.x
pip install whispersx



2.3

python3 -m venv whisperx_env && source whisperx_env/bin/activate && pip install whisperx && pip install torch torchvision torchaudio --index-url

pip install ctranslate2==4.5.0