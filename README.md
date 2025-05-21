# DMDC README
*WIP README*
## Project Description

The goal of this project is to reproduce the data creation process described in the *RealTalk* paper to extract dyadic multimodal conversations from platforms such as YouTube, Spotify, and Dailymotion. This work is part of a broader research initiative aimed at building listening agents. The resulting dataset will also be reusable for other projects related to social interaction and multimodal AI.



## Installation
1. Clone this repo  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
   Run the main script:
   ```bash
   python main.py
   ```
    
3. It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

   ```bash
   # on Ubuntu or Debian
   sudo apt update && sudo apt install ffmpeg

   # on Arch Linux
   sudo pacman -S ffmpeg

   # on MacOS using Homebrew (https://brew.sh/)
   brew install ffmpeg

   # on Windows using Chocolatey (https://chocolatey.org/)
   choco install ffmpeg

   # on Windows using Scoop (https://scoop.sh/)
   scoop install ffmpeg
   ```