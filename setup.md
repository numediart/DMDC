# Setup DMDC for Linux

## Needed
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [FFmpeg](https://ffmpeg.org/download.html#build-linux)

## Setup of the conda environment.
   ```bash
      conda env create -f environment.yml
   ```
## Delete the conda env
   ```bash
   conda env remove -n dmdc-pipeline
   ```


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