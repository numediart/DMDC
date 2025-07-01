import whisperx
import gc
import pandas as pd
import os

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def whisperX_process(audio_file, output_folder="./output", batch_size=16, device="cuda", compute_type="int8"):

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # gc.collect()

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(result["segments"])  # after alignment

    # delete model if low on GPU resources
    gc.collect()

    # 3. Assign speaker labels
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio)
    # diarize_segments =diarize_model(audio, min_speakers=0, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"])  # segments are now assigned speaker IDs

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert segments to DataFrame
    segments_df = pd.DataFrame(result["segments"])

    # Export to JSON
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file_json = os.path.join(output_folder, f"{base_name}_diarization.json")
    segments_df.to_json(output_file_json, orient="records", lines=True)

    print(f"Diarization saved to {output_file_json}")


if __name__ == "__main__":
    audio_file = os.path.join(os.path.dirname(__file__), "input", "7_video.wav")
    whisperX_process(audio_file)
