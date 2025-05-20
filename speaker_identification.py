
import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import sounddevice as sd
import scipy.io.wavfile as wav
import pickle
import random
import uuid
from pathlib import Path

# Define basic parameters
AUDIO_DIR = r"sounds_arabic"
FEATURES_DIR = r"features"
MODEL_DIR = r"models"

N_MFCC = 40
N_COMPONENTS = 256
PCA_COMPONENTS = 30
THRESHOLD = 60
BATCH_SIZE = 100000
WEIGHT_MFCC = 0.7
WEIGHT_DELTA = 0.2
WEIGHT_DELTA_DELTA = 0.1
VAD_TOP_DB = 30  # Threshold for voice activity detection (in decibels)
VAD_FRAME_LENGTH = 2048  # Frame length for VAD
VAD_HOP_LENGTH = 512  # Hop length for VAD

# Create directories if they don't exist
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Spectral subtraction for noise reduction with adjustable parameters
def spectral_subtraction(audio, sr, noise_duration=0.5, n_fft=2048, hop_length=512, over_subtraction=1.0):
    """
    Perform spectral subtraction to reduce noise.
    
    Parameters:
    - audio: Input audio signal
    - sr: Sampling rate
    - noise_duration: Duration (in seconds) of noise sample to estimate noise profile
    - n_fft: FFT window size
    - hop_length: Hop length for STFT
    - over_subtraction: Factor to control strength of noise subtraction (1.0 is standard)
    
    Returns:
    - audio_denoised: Denoised audio signal
    """
    try:
        # Estimate noise profile from the initial portion
        noise_sample = audio[:int(noise_duration * sr)]
        if len(noise_sample) < n_fft:
            # Pad noise sample if too short
            noise_sample = np.pad(noise_sample, (0, n_fft - len(noise_sample)), mode='constant')
        
        # Compute STFT for audio and noise
        stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_noise = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)
        
        # Magnitude and phase
        mag_audio, phase_audio = np.abs(stft_audio), np.angle(stft_audio)
        mag_noise = np.abs(stft_noise)
        
        # Use median for more robust noise estimation
        noise_median = np.median(mag_noise, axis=1, keepdims=True)
        
        # Apply over-subtraction
        mag_denoised = np.maximum(mag_audio - over_subtraction * noise_median, 0.0)
        
        # Reconstruct signal
        stft_denoised = mag_denoised * np.exp(1j * phase_audio)
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length, length=len(audio))
        
        return audio_denoised
    except Exception as e:
        print(f"Error in spectral subtraction: {e}")
        return audio  # Return original audio if processing fails

# Extract features with improved VAD, noise reduction, and data augmentation
def extract_features(file_path=None, audio=None, sr=None, n_mfcc=40, augment=True):
    try:
        if file_path:
            y, sr = librosa.load(file_path, sr=None)
        else:
            y = audio
        
        # Estimate noise level and adjust top_db dynamically
        noise_level = np.max(librosa.feature.rms(y=y)) if np.max(y) != 0 else 1e-6
        dynamic_top_db = max(20, min(40, 30 + 10 * np.log10(noise_level)))  # Dynamic adjustment
        
        # Apply Voice Activity Detection (VAD) with dynamic top_db
        non_silent_intervals = librosa.effects.split(
            y, top_db=dynamic_top_db, frame_length=2048, hop_length=512
        )
        if len(non_silent_intervals) == 0:
            print("Warning: No speech detected after VAD, using entire audio.")
            y_speech = y
        else:
            # Concatenate non-silent segments
            y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        
        # Apply noise reduction
        y_speech = spectral_subtraction(
            y_speech, sr, noise_duration=0.5, n_fft=2048, hop_length=512, over_subtraction=1.2
        )
        
        augmented_features = []
        if augment:
            y_original = y_speech
            pitch_shift = random.uniform(-2, 2)
            y_pitch = librosa.effects.pitch_shift(y_speech, sr=sr, n_steps=pitch_shift)
            time_stretch = random.uniform(0.8, 1.2)
            y_stretch = librosa.effects.time_stretch(y_speech, rate=time_stretch)
            
            for audio_var in [y_original, y_pitch, y_stretch]:
                audio_var = librosa.effects.preemphasis(audio_var)
                mfccs = librosa.feature.mfcc(y=audio_var, sr=sr, n_mfcc=n_mfcc)
                delta_mfccs = librosa.feature.delta(mfccs)
                delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
                if mfccs.size == 0 or delta_mfccs.size == 0 or delta_delta_mfccs.size == 0:
                    print(f"Warning: Empty feature matrix for augmented audio")
                    continue
                augmented_features.append((mfccs.T, delta_mfccs.T, delta_delta_mfccs.T))
            
            if not augmented_features:
                print("Error: No valid augmented features extracted")
                return None, None, None
                
            mfccs = np.vstack([feat[0] for feat in augmented_features])
            delta_mfccs = np.vstack([feat[1] for feat in augmented_features])
            delta_delta_mfccs = np.vstack([feat[2] for feat in augmented_features])
        else:
            y_speech = librosa.effects.preemphasis(y_speech)
            mfccs = librosa.feature.mfcc(y=y_speech, sr=sr, n_mfcc=n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
            if mfccs.size == 0 or delta_mfccs.size == 0 or delta_delta_mfccs.size == 0:
                print("Error: Empty feature matrix for non-augmented audio")
                return None, None, None
            mfccs, delta_mfccs, delta_delta_mfccs = mfccs.T, delta_mfccs.T, delta_delta_mfccs.T
        
        return mfccs, delta_mfccs, delta_delta_mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None

# Apply CMVN using NumPy for CPU
def apply_cmvn(features):
    features = np.asarray(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized = (features - mean) / (std + 1e-6)
    return normalized

# Train Universal Background Model (UBM) with batch processing
def train_ubm(features_dir, individuals):
    print("\nTraining UBM...")
    all_features = []
    total_frames = 0
    for person in individuals:
        for i in range(1, 56):
            try:
                mfccs = apply_cmvn(np.load(os.path.join(features_dir, f"{person}{i}_mfccs.npy")))
                delta_mfccs = apply_cmvn(np.load(os.path.join(features_dir, f"{person}{i}_delta_mfccs.npy")))
                delta_delta_mfccs = apply_cmvn(np.load(os.path.join(features_dir, f"{person}{i}_delta_delta_mfccs.npy")))
                mfccs = np.asarray(mfccs)
                delta_mfccs = np.asarray(delta_mfccs)
                delta_delta_mfccs = np.asarray(delta_delta_mfccs)
                weighted_mfccs = WEIGHT_MFCC * mfccs
                weighted_delta = WEIGHT_DELTA * delta_mfccs
                weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
                combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
                all_features.append(combined_features)
                total_frames += combined_features.shape[0]
                if total_frames >= BATCH_SIZE:
                    all_features = np.vstack(all_features)
                    yield all_features
                    all_features = []
                    total_frames = 0
            except FileNotFoundError:
                print(f"Feature files missing for {person}{i}. Skipping.")
                continue
    if all_features:
        all_features = np.vstack(all_features)
        yield all_features

# Fit UBM incrementally
def fit_ubm_incremental(features_dir, individuals):
    pca = PCA(n_components=PCA_COMPONENTS)
    ubm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag', max_iter=200, random_state=42)
    first_batch = True
    for batch in train_ubm(features_dir, individuals):
        batch_pca = pca.fit_transform(batch) if first_batch else pca.transform(batch)
        if first_batch:
            ubm.fit(batch_pca)
            first_batch = False
        else:
            ubm.fit(batch_pca)
        print(f"Processed batch with {batch.shape[0]} frames.")
    print("UBM trained with total frames processed.")
    return ubm, pca

# Train GMM models for each speaker
def train_gmm_models(ubm, pca, features_dir):
    gmm_models = {}
    # Get list of unique speaker IDs from feature files
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_mfccs.npy')]
    speakers = set()
    for f in feature_files:
        speaker_id = f.split('_')[0] if '_' in f else f.replace('_mfccs.npy', '')
        speakers.add(speaker_id)
    
    for person in speakers:
        print(f"\nTraining GMM for {person} using MAP adaptation...")
        all_features = []
        for f in feature_files:
            if f.startswith(f"{person}_") and f.endswith("_mfccs.npy"):
                try:
                    base_name = f.replace("_mfccs.npy", "")
                    mfccs = apply_cmvn(np.load(os.path.join(features_dir, f)))
                    delta_mfccs = apply_cmvn(np.load(os.path.join(features_dir, f"{base_name}_delta_mfccs.npy")))
                    delta_delta_mfccs = apply_cmvn(np.load(os.path.join(features_dir, f"{base_name}_delta_delta_mfccs.npy")))
                    mfccs = np.asarray(mfccs)
                    delta_mfccs = np.asarray(delta_mfccs)
                    delta_delta_mfccs = np.asarray(delta_delta_mfccs)
                    weighted_mfccs = WEIGHT_MFCC * mfccs
                    weighted_delta = WEIGHT_DELTA * delta_mfccs
                    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
                    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
                    combined_features_pca = pca.transform(combined_features)
                    all_features.append(combined_features_pca)
                except FileNotFoundError:
                    print(f"Feature files missing for {f}. Skipping.")
                    continue
        if all_features:
            all_features = np.vstack(all_features)
            gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag', max_iter=200, random_state=42)
            gmm.means_ = ubm.means_
            gmm.covariances_ = ubm.covariances_
            gmm.weights_ = ubm.weights_
            gmm.fit(all_features)
            gmm_models[person] = gmm
            print(f"GMM trained for {person} with {len(all_features)} frames.")
    return gmm_models

# Save models
def save_models(ubm, pca, gmm_models):
    ubm_path = os.path.join(MODEL_DIR, "ubm.pkl")
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    gmm_path = os.path.join(MODEL_DIR, "gmm_models.pkl")
    with open(ubm_path, 'wb') as f:
        pickle.dump(ubm, f)
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    with open(gmm_path, 'wb') as f:
        pickle.dump(gmm_models, f)
    print("Models saved successfully.")

# Load models
def load_models():
    ubm_path = os.path.join(MODEL_DIR, "ubm.pkl")
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    gmm_path = os.path.join(MODEL_DIR, "gmm_models.pkl")
    try:
        with open(ubm_path, 'rb') as f:
            ubm = pickle.load(f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        with open(gmm_path, 'rb') as f:
            gmm_models = pickle.load(f)
        print("Models loaded successfully.")
        return ubm, pca, gmm_models
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Record audio
def record_audio(duration=15, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    return recording.flatten(), fs

# Load audio from file
def load_audio_from_file(file_path):
    print(f"Loading audio from {file_path}...")
    try:
        sr, audio = wav.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        print("Audio loaded successfully.")
        return audio, sr
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the path.")
        return None, None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

# Register new speaker
def register_speaker(speaker_id, audio_list, sr_list, ubm, pca, gmm_models):
    if speaker_id in gmm_models:
        print(f"Error: Speaker ID '{speaker_id}' already exists.")
        return False
    
    print(f"Registering new speaker: {speaker_id}")
    all_features = []
    for audio, sr in zip(audio_list, sr_list):
        audio = audio.astype(float) / 32768.0
        mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio, sr=sr, augment=True)
        if mfccs is None:
            print("Failed to extract features for one of the files.")
            continue
        
        # Save features
        unique_id = str(uuid.uuid4())[:8]
        np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_mfccs.npy"), mfccs)
        np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_delta_mfccs.npy"), delta_mfccs)
        np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_delta_delta_mfccs.npy"), delta_delta_mfccs)
        
        # Apply CMVN and combine features
        mfccs = apply_cmvn(mfccs)
        delta_mfccs = apply_cmvn(delta_mfccs)
        delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
        weighted_mfccs = WEIGHT_MFCC * mfccs
        weighted_delta = WEIGHT_DELTA * delta_mfccs
        weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
        combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
        combined_features_pca = pca.transform(combined_features)
        all_features.append(combined_features_pca)
    
    if not all_features:
        print("Failed to extract features for all files.")
        return False
    
    # Train GMM for new speaker
    all_features = np.vstack(all_features)
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag', max_iter=200, random_state=42)
    gmm.means_ = ubm.means_
    gmm.covariances_ = ubm.covariances_
    gmm.weights_ = ubm.weights_
    gmm.fit(all_features)
    
    # Update models
    gmm_models[speaker_id] = gmm
    
    # Save updated models
    save_models(ubm, pca, gmm_models)
    print(f"Speaker {speaker_id} registered successfully.")
    return True

# Identify speaker
def identify_speaker(audio, sr, gmm_models, ubm, pca, threshold):
    if audio is None or sr is None:
        print("Cannot identify speaker: Invalid audio data.")
        return
    audio = audio.astype(float) / 32768.0
    mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio, sr=sr, augment=False)
    if mfccs is None:
        print("Failed to extract features for identification.")
        return
    mfccs = apply_cmvn(mfccs)
    delta_mfccs = apply_cmvn(delta_mfccs)
    delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
    mfccs = np.asarray(mfccs)
    delta_mfccs = np.asarray(delta_mfccs)
    delta_delta_mfccs = np.asarray(delta_delta_mfccs)
    weighted_mfccs = WEIGHT_MFCC * mfccs
    weighted_delta = WEIGHT_DELTA * delta_mfccs
    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
    combined_features_pca = pca.transform(combined_features)
    likelihood_ratios = {}
    for person, gmm in gmm_models.items():
        score_target = gmm.score(combined_features_pca)
        score_ubm = ubm.score(combined_features_pca)
        likelihood_ratio = score_target - score_ubm
        likelihood_ratios[person] = likelihood_ratio
    min_similarity = -10
    max_similarity = 10
    percentages = {person: max(0, min(100, 100 * (lr - min_similarity) / (max_similarity - min_similarity))) 
                   for person, lr in likelihood_ratios.items()}
    best_speaker = max(percentages, key=percentages.get)
    best_percentage = percentages[best_speaker]
    print(f"The voice resembles {best_speaker} with a similarity of {best_percentage:.2f}%")
    if best_percentage < threshold:
        print(f"(Note: Similarity is below the threshold of {threshold}%, confidence is low)")

# Verify speaker
def verify_speaker(audio, sr, claimed_identity, gmm_models, ubm, pca, threshold):
    if claimed_identity not in gmm_models:
        print(f"Error: {claimed_identity} is not a registered speaker.")
        return
    if audio is None or sr is None:
        print("Cannot verify speaker: Invalid audio data.")
        return
    audio = audio.astype(float) / 32768.0
    mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio, sr=sr, augment=False)
    if mfccs is None:
        print("Failed to extract features for verification.")
        return
    mfccs = apply_cmvn(mfccs)
    delta_mfccs = apply_cmvn(delta_mfccs)
    delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
    mfccs = np.asarray(mfccs)
    delta_mfccs = np.asarray(delta_mfccs)
    delta_delta_mfccs = np.asarray(delta_delta_mfccs)
    weighted_mfccs = WEIGHT_MFCC * mfccs
    weighted_delta = WEIGHT_DELTA * delta_mfccs
    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
    combined_features_pca = pca.transform(combined_features)
    gmm = gmm_models[claimed_identity]
    score_target = gmm.score(combined_features_pca)
    score_ubm = ubm.score(combined_features_pca)
    likelihood_ratio = score_target - score_ubm
    min_similarity = -10
    max_similarity = 10
    percentage = max(0, min(100, 100 * (likelihood_ratio - min_similarity) / (max_similarity - min_similarity)))
    if percentage >= threshold:
        print(f"Verification successful: The speaker is {claimed_identity} with a similarity of {percentage:.2f}%")
    else:
        print(f"Verification failed: The speaker is not {claimed_identity} (similarity: {percentage:.2f}%)")

# Delete speaker
def delete_speaker(speaker_id, gmm_models, ubm, pca):
    if speaker_id not in gmm_models:
        print(f"Error: Speaker ID '{speaker_id}' does not exist.")
        return False
    
    print(f"Deleting speaker: {speaker_id}")
    # Remove speaker from gmm_models
    del gmm_models[speaker_id]
    
    # Delete feature files
    feature_files = [
        f for f in os.listdir(FEATURES_DIR)
        if f.startswith(f"{speaker_id}_") and f.endswith(("_mfccs.npy", "_delta_mfccs.npy", "_delta_delta_mfccs.npy"))
    ]
    for file in feature_files:
        try:
            os.remove(os.path.join(FEATURES_DIR, file))
            print(f"Deleted feature file: {file}")
        except Exception as e:
            print(f"Error deleting feature file {file}: {e}")
    
    # Save updated models
    save_models(ubm, pca, gmm_models)
    print(f"Speaker {speaker_id} deleted successfully.")
    return True

# Main execution
if __name__ == "__main__":
    print("Speaker Identification and Verification System (Running on CPU)")
    while True:
        print("\nChoose an option:")
        print("1. Extract features")
        print("2. Train models")
        print("3. Identify speaker (record audio)")
        print("4. Identify speaker (load audio file)")
        print("5. Verify speaker (record audio)")
        print("6. Verify speaker (load audio file)")
        print("7. Register new speaker (record audio)")
        print("8. Register new speaker (load audio file)")
        print("9. Delete speaker")
        print("10. Exit")
        choice = input("Enter your choice (1-10): ")

        if choice == '1':
            print("Extracting features...")
            feature_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
            for file_name in feature_files:
                file_path = os.path.join(AUDIO_DIR, file_name)
                speaker_id = file_name.split('_')[0] if '_' in file_name else file_name.replace('.wav', '')
                mfccs, delta_mfccs, delta_delta_mfccs = extract_features(file_path=file_path, augment=True)
                if mfccs is not None:
                    unique_id = str(uuid.uuid4())[:8]
                    np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_mfccs.npy"), mfccs)
                    np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_delta_mfccs.npy"), delta_mfccs)
                    np.save(os.path.join(FEATURES_DIR, f"{speaker_id}_{unique_id}_delta_delta_mfccs.npy"), delta_delta_mfccs)
                    print(f"Extracted and saved features for {file_name}")
                else:
                    print(f"Failed to extract features for {file_name}")
            print("Feature extraction complete!")
        elif choice == '2':
            feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('_mfccs.npy')]
            speakers = set(f.split('_')[0] for f in feature_files)
            if not speakers:
                print("No feature files found. Please extract features first.")
                continue
            ubm, pca = fit_ubm_incremental(FEATURES_DIR, speakers)
            if ubm and pca:
                gmm_models = train_gmm_models(ubm, pca, FEATURES_DIR)
                save_models(ubm, pca, gmm_models)
            else:
                print("Failed to train UBM. Please check feature files.")
        elif choice == '3':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                audio, sr = record_audio(duration=15, fs=16000)
                wav.write("new_recording.wav", sr, audio)
                identify_speaker(audio, sr, gmm_models, ubm, pca, THRESHOLD)
            else:
                print("Models not found. Please train models first.")
        elif choice == '4':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                file_path = input("Enter the path to the audio file (e.g., sounds_arabic/sample1.wav): ")
                file_path = os.path.normpath(file_path.strip('"').strip("'"))
                audio, sr = load_audio_from_file(file_path)
                identify_speaker(audio, sr, gmm_models, ubm, pca, THRESHOLD)
            else:
                print("Models not found. Please train models first.")
        elif choice == '5':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                claimed_identity = input("Enter the claimed speaker's name: ").strip()
                audio, sr = record_audio(duration=15, fs=16000)
                wav.write("new_recording.wav", sr, audio)
                verify_speaker(audio, sr, claimed_identity, gmm_models, ubm, pca, THRESHOLD)
            else:
                print("Models not found. Please train models first.")
        elif choice == '6':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                claimed_identity = input("Enter the claimed speaker's name: ").strip()
                file_path = input("Enter the path to the audio file (e.g., sounds_arabic/sample1.wav): ")
                file_path = os.path.normpath(file_path.strip('"').strip("'"))
                audio, sr = load_audio_from_file(file_path)
                verify_speaker(audio, sr, claimed_identity, gmm_models, ubm, pca, THRESHOLD)
            else:
                print("Models not found. Please train models first.")
        elif choice == '7':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                speaker_id = input("Enter the new speaker's ID: ").strip()
                num_recordings = int(input("How many audio recordings do you want to register? "))
                audio_list = []
                sr_list = []
                for i in range(num_recordings):
                    print(f"Preparing to record audio {i+1}...")
                    audio, sr = record_audio(duration=15, fs=16000)
                    wav.write(f"new_speaker_recording_{i+1}.wav", sr, audio)
                    audio_list.append(audio)
                    sr_list.append(sr)
                if audio_list:
                    register_speaker(speaker_id, audio_list, sr_list, ubm, pca, gmm_models)
            else:
                print("Models not found. Please train models first.")
        elif choice == '8':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                speaker_id = input("Enter the new speaker's ID: ").strip()
                num_files = int(input("How many audio files do you want to register? "))
                audio_list = []
                sr_list = []
                for i in range(num_files):
                    file_path = input(f"Enter the path to audio file {i+1} (e.g., sounds_arabic/sample1.wav): ")
                    file_path = os.path.normpath(file_path.strip('"').strip("'"))
                    audio, sr = load_audio_from_file(file_path)
                    if audio is not None and sr is not None:
                        audio_list.append(audio)
                        sr_list.append(sr)
                if audio_list:
                    register_speaker(speaker_id, audio_list, sr_list, ubm, pca, gmm_models)
            else:
                print("Models not found. Please train models first.")
        elif choice == '9':
            ubm, pca, gmm_models = load_models()
            if ubm and pca and gmm_models:
                speaker_id = input("Enter the speaker's ID to delete: ").strip()
                delete_speaker(speaker_id, gmm_models, ubm, pca)
            else:
                print("Models not found. Please train models first.")
        elif choice == '10':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10.")

print("Program terminated.")
