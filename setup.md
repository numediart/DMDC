# Setup DMDC for Linux

## Needed

- [Docker](https://docs.docker.com/engine/install/ubuntu/):  
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   ```

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):  
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

- [FFmpeg](https://ffmpeg.org/download.html#build-linux):  
   ```bash
   sudo apt update
   sudo apt install -y ffmpeg
   ```

## Setup of the conda environment.
   ```bash
      conda env create -f environment.yml
   ```
## Delete the conda env
   ```bash
      conda env remove -n dmdc-pipeline
   ```

# Additional Setups

## Additional Setup for Parakeet transcription

         pip install -r requirements_parakeet.txt
         pip install parakeet
         ```

      1. Create a virtual environment:
         ```bash
         python3.12 -m venv .venv_parakeet
         ```
      2. Activate the virtual environment:
         ```bash
         .venv_parakeet\Scripts\activate
         ```
      3. Install required packages:
         ```bash
         pip install -r requirements_parakeet.txt
         pip install parakeet
         ```

## Additional Setup for face croping


         ```bash
         conda activate dmdc-pipeline
         pip install git+https://github.com/deepinsight/insightface.git
         ```
