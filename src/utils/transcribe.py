import os
import whisper
from pathlib import Path
import torch

def transcribe_audio_files(data_folder=None, model_size="large"):
    """
    Transcribe clean audio files from Task1 and Task2 using Whisper.
    """

    # auto-detect dataset folder
    if data_folder is None:
        data_folder = Path(__file__).resolve().parents[2] / "dataset"

    data_folder = Path(data_folder)
    print("Using dataset path:", data_folder)
    
    # Load Whisper model
    print(f"Loading Whisper {model_size} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    print("Running on:", device)
    
    # Define tasks and splits
    tasks = ["Task1", "Task2"]
    splits = ["train", "val"]
    
    for task in tasks:
        for split in splits:
            # Define paths
            clean_audio_path = Path(data_folder) / task / "Clean" / split
            transcription_output_path = Path(data_folder) / "transcriptions" / task / split
            
            # Create output directory if it doesn't exist
            transcription_output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if clean audio folder exists
            if not clean_audio_path.exists():
                print(f"Warning: {clean_audio_path} does not exist. Skipping {task}/{split}.")
                continue
            
            # Get all audio files (common formats)
            audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(clean_audio_path.glob(f"*{ext}"))
            
            # Sort to maintain consistent order
            audio_files = sorted(audio_files)
            
            if not audio_files:
                print(f"No audio files found in {clean_audio_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {task}/{split}: Found {len(audio_files)} audio files")
            print(f"{'='*60}")
            
            # Transcribe each audio file
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\n[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                
                try:
                    # Transcribe
                    result = model.transcribe(str(audio_file))
                    
                    # Create output filename (same name but .txt extension)
                    output_file = transcription_output_path / f"{audio_file.stem}.txt"
                    
                    # Save transcription
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result["text"])
                    
                    print(f"✓ Saved to: {output_file}")
                    print(f"  Preview: {result['text'][:100]}...")
                    
                except Exception as e:
                    print(f"✗ Error transcribing {audio_file.name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Transcription complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # You can change the model size here: tiny, base, small, medium, large
    # Larger models are more accurate but slower
    # base is a good balance for most use cases
    
    transcribe_audio_files(
        data_folder="../dataset",
        model_size="large"  # Change to "small", "medium", or "large" for better accuracy
    )