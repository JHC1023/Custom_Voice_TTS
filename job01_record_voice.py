import csv
import os
import time
import sounddevice as sd
import wavio
import speech_recognition as sr
import keyboard
import Levenshtein
import numpy as np
from datetime import datetime
from playsound import playsound

# File paths
CSV_FILE = "korean_corpus.csv"
PROGRESS_FILE = "recording_progress.txt"
OUTPUT_DIR = "recordings"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Recording parameters
SAMPLE_RATE = 44100  # Hz
RECORD_KEY = 'space'  # Key to start/stop recording


def load_sentences():
    """Load Korean sentences from the CSV file."""
    sentences = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Ensure row is not empty
                sentences.append(row[0])  # First column is the sentence
    return sentences


def load_progress():
    """Load the last recorded sentence index from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_progress(index):
    """Save the current sentence index to the progress file."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))


def record_audio(filename):
    """Record audio when the RECORD_KEY is pressed and released."""
    print(f"Press and hold '{RECORD_KEY}' to start recording, release to stop...")
    # Wait for key press
    keyboard.wait(RECORD_KEY)
    print("Recording started...")
    recording = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
        while keyboard.is_pressed(RECORD_KEY):
            data, _ = stream.read(1024)
            recording.append(data)
        print("Recording stopped.")
    # Convert list to numpy array and save as WAV
    recording = np.concatenate(recording, axis=0)
    wavio.write(filename, recording, SAMPLE_RATE, sampwidth=2)
    return filename


def speech_to_text(audio_file):
    """Convert recorded audio to text using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        # Use Google Speech Recognition with Korean language
        text = recognizer.recognize_google(audio, language='ko-KR')
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return ""


def calculate_accuracy(original, transcribed):
    """Calculate accuracy between original and transcribed text using Levenshtein distance."""
    if not transcribed:
        return 0.0
    distance = Levenshtein.distance(original, transcribed)
    max_length = max(len(original), len(transcribed))
    if max_length == 0:
        return 100.0 if original == transcribed else 0.0
    accuracy = (1 - distance / max_length) * 100
    return max(0.0, accuracy)


def main():
    # Load sentences and progress
    sentences = load_sentences()
    total_sentences = len(sentences)
    current_index = load_progress()

    if current_index >= total_sentences:
        print("All sentences have been recorded!")
        return

    recognizer = sr.Recognizer()

    while current_index < total_sentences:
        sentence = sentences[current_index]
        print(f"\nSentence {current_index + 1}/{total_sentences}: {sentence}")

        # Generate unique filename for the recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join(OUTPUT_DIR, f"sentence_{current_index + 1}_{timestamp}.wav")

        while True:
            # Record audio
            record_audio(audio_file)

            # Convert to text and calculate accuracy
            transcribed_text = speech_to_text(audio_file)
            accuracy = calculate_accuracy(sentence, transcribed_text)
            print(f"\nTranscribed: {transcribed_text}")
            print(f"Accuracy: {accuracy:.2f}%")

            # Present options
            print("\nOptions:")
            print("1. Play recording")
            print("2. Re-record")
            print("3. Next sentence")
            print("4. Exit")
            choice = input("Enter choice (1-4): ").strip()

            if choice == '1':
                try:
                    playsound(audio_file)
                except Exception as e:
                    print(f"Error playing audio: {e}")
            elif choice == '2':
                # Delete the current recording and re-record
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                continue
            elif choice == '3':
                # Move to next sentence
                save_progress(current_index + 1)
                current_index += 1
                break
            elif choice == '4':
                # Exit and save progress
                save_progress(current_index)
                print("Exiting and saving progress...")
                return
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")

        if current_index >= total_sentences:
            print("All sentences have been recorded!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Progress saved.")
    except Exception as e:
        print(f"An error occurred: {e}")