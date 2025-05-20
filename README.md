Speaker Identification and Verification System
This project implements a speaker identification and verification system using Gaussian Mixture Models (GMM) with Mel-Frequency Cepstral Coefficients (MFCC) features. It includes voice activity detection (VAD), noise reduction via spectral subtraction, and data augmentation for robust speaker modeling.
Features

Feature Extraction: Extracts MFCC, delta, and delta-delta features from audio files.
Model Training: Trains a Universal Background Model (UBM) and speaker-specific GMMs.
Speaker Identification: Identifies speakers from recorded or file-based audio.
Speaker Verification: Verifies a claimed speaker identity.
Speaker Registration: Registers new speakers with a single audio file (minimum 60 seconds).
Speaker Deletion: Removes registered speakers and their associated data.

Prerequisites

Python: Version 3.8 or higher.
Operating System: Windows, macOS, or Linux.
Hardware: Microphone for recording audio (optional for file-based operations).
Dependencies: Listed in requirements.txt.

Installation

Clone the Repository:
git clone https://github.com/faiha31/SpeakerIdentificationSystem.git
cd SpeakerIdentificationSystem


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare Directory Structure:The following directories will be created automatically by the script if not present:

sounds_arabic/: Contains sample audio files (Dareen1.wav, Faihaa1.wav) for testing.
features/: For storing extracted feature files (.npy format).
models/: For storing trained model files (.pkl format).

To create them manually:
mkdir features models


Prepare Audio Files (for testing):

The repository includes two sample WAV files in sounds_arabic/:
Dareen1.wav: Sample audio for speaker "Dareen1".
Faihaa1.wav: Sample audio for speaker "Faihaa1".


For full training, place additional audio files in sounds_arabic/ with the naming convention <speaker_name><number>.wav (e.g., Dareen2.wav, Faihaa2.wav).
Ensure audio files are in WAV format and have sufficient quality (e.g., 16kHz sampling rate).



Usage

Run the Script:
python speaker_identification.py


Menu Options:The script provides an interactive menu with the following options:

1. Extract features: Extracts MFCC features from audio files in sounds_arabic/.
2. Train models: Trains the UBM and GMMs using extracted features.
3. Identify speaker (record audio): Records 15 seconds of audio and identifies the speaker.
4. Identify speaker (load audio file): Identifies the speaker from a provided WAV file.
5. Verify speaker (record audio): Records 15 seconds of audio and verifies against a claimed identity.
6. Verify speaker (load audio file): Verifies the speaker from a provided WAV file.
7. Register new speaker (record audio): Records a 60-second audio to register a new speaker.
8. Register new speaker (load audio file): Registers a new speaker using a WAV file (minimum 60 seconds).
9. Delete speaker: Deletes a registered speaker and their associated data.
10. Exit: Terminates the program.


Testing with Sample Audio Files:

Train Models:
Select option 1 to extract features from sounds_arabic/Dareen2.wav and sounds_arabic/Faihaa2.wav.
Select option 2 to train the UBM and GMMs for "Dareen" and "Faihaa".


Identify Speaker:
Select option 4 and enter the path sounds_arabic/Dareen1.wav or sounds_arabic/Faihaa1.wav.
Example output: The voice resembles Faihaa with a similarity of 85.77%.


Verify Speaker:
Select option 6, enter the claimed speaker’s name (e.g., Dareen), and provide the path sounds_arabic/Dareen1.wav.
Example output: Verification successful: The speaker is Dareen with a similarity of 95.63%.


Note: The sample files are for testing only. For accurate results, train the system with additional audio files for each speaker.



Notes

Audio Requirements:
Audio files must be in WAV format.
For registration, audio must be at least 60 seconds long.
For identification/verification, audio should be at least 15 seconds for best results.


Sample Audio Files:
The included Dareen1.wav and Faihaa1.wav are pre-trained in the GMM models (if you follow the training steps).
Use these files to test options 4 and 6 without needing to record new audio.


Performance:
The system uses CPU-based processing for feature extraction and model training.
Large datasets may require significant memory and processing time.


Troubleshooting:
Ensure all dependencies are installed correctly.
Verify that audio files are accessible and in the correct format.
Check that the directories (sounds_arabic/, features/, models/) are writable.
If models fail to load, ensure you’ve run options 1 and 2 to generate features and models.


Limitations:
The system assumes a predefined list of speakers in the individuals list in the script.
Modify the individuals list in speaker_identification.py if you add new speakers manually.



Directory Structure
SpeakerIdentificationSystem/
├── speaker_identification.py  # Main Python script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── README.md                # This file
├── sounds_arabic/           # Directory with sample audio files
│   ├── Dareen1.wav
│   ├── Faihaa1.wav
├── features/     # Directory for feature files (not in repo)
├── models/       # Directory for model files ( in repo)

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports or feature requests.
Contact
For questions or support, please open an issue on the GitHub repository.
"# SpeakerIdentificationSystem" 
"# SpeakerIdentificationSystem" 
