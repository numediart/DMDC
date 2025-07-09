# DMDC README
*WIP README*

## Project Description

The goal of this project is to reproduce the data creation process described in the *RealTalk* paper to extract dyadic multimodal conversations from platforms such as YouTube, Spotify, and Dailymotion. This work is part of a broader research initiative aimed at building listening agents. The resulting dataset will also be reusable for other projects related to social interaction and multimodal AI.


## Installation
1. **Clone this repo**  
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

    
3. **FFMPEG**  
It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

   ```bash
   # on Ubuntu or Debian
   sudo apt update && sudo apt install ffmpeg

   # on Windows
   choco install ffmpeg  #  Chocolatey (https://chocolatey.org/)
   scoop install ffmpeg   #  Scoop (https://scoop.sh/)
   ```
4. **Additional Setup for Parakeet transcription**

         pip install -r requirements_parakeet.txt
         pip install parakeet
         ```

      **Windows**
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