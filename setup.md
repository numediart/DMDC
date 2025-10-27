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
   # Add Miniconda to PATH and create a global "conda" command
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   # Remove the installer after installation (optional)
   rm Miniconda3-latest-Linux-x86_64.sh
   ```

- [FFmpeg](https://ffmpeg.org/download.html#build-linux):  
   ```bash
   sudo apt update
   sudo apt install -y ffmpeg
   ```

> **Note:** Restart your PC after completing the installations above to ensure all changes take effect.

## Setup of the conda environment.
   ```bash
      conda env create -f environment.yml
   ```
## Delete the conda env
   ```bash
      conda env remove -n dmdc-pipeline
   ```
## Install/Run the openface container (Size: 10Gb)
   ```bash
   docker run -it --rm algebr/openface:latest
   #OR
   sudo docker run -it --rm algebr/openface:latest
   ```

# Additional Setups

## Additional Setup for Parakeet transcription

   1. Create a virtual environment with conda:
      ```bash
         1. Create a virtual environment:
      ```
   2. Activate the virtual environment:
      ```bash
      conda activate parakeet
      ```
   3. Once activated, install the required packages:
      ```bash
      pip install -r requirements_parakeet.txt
      pip install parakeet-asr
      ```

## Additional Setup for face croping


         ```bash
         conda activate dmdc-pipeline
         pip install git+https://github.com/deepinsight/insightface.git
         ```
