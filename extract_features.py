#feature extraction.py
import parselmouth
import librosa
import numpy as np

def extract_features(path):
    sound = parselmouth.Sound(path)
    
    #average pitch extraction
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_mean = np.mean(pitch_values[pitch_values > 0])
    pitch_std  = np.std(pitch_values[pitch_values > 0])
    
    #jitter/shimmer extraction
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)",0 , 0 , 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Formant extraction
    formant = sound.to_formant_burg(time_step=0.01)
    times = [formant.get_time_from_frame_number(i) for i in range(1, formant.get_number_of_frames() + 1)]
    
    f1_vals = [formant.get_value_at_time(1, t) for t in times if formant.get_value_at_time(1, t) is not None]
    f2_vals = [formant.get_value_at_time(2, t) for t in times if formant.get_value_at_time(2, t) is not None]
    f3_vals = [formant.get_value_at_time(3, t) for t in times if formant.get_value_at_time(3, t) is not None]
    
    f1_mean = np.mean(f1_vals)
    f1_std  = np.std(f1_vals)
    f1_min  = np.min(f1_vals)
    f1_max  = np.max(f1_vals)
    f1_range = f1_max - f1_min
    
    f2_mean = np.mean(f2_vals)
    f2_std  = np.std(f2_vals)
    f2_min  = np.min(f2_vals)
    f2_max  = np.max(f2_vals)
    f2_range = f2_max - f2_min
    
    f3_mean = np.mean(f3_vals)
    f3_std  = np.std(f3_vals)
    f3_min  = np.min(f3_vals)
    f3_max  = np.max(f3_vals)
    f3_range = f3_max - f3_min
    
    f2_f1_ratio_mean = np.mean(np.array(f2_vals) / np.array(f1_vals))
    
    #extracting harmocity
    harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_values = harmonicity.values[0] #first row of the harmonicity matrix
    hnr_mean = np.mean(hnr_values[hnr_values != -200]) #excluding praats -200 decible floor cutoff
    hnr_std = np.std(hnr_values[hnr_values != -200]) #doing the same
    
    #MFCC mean extraction(using librosa)
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc1_mean = np.mean(mfcc[0])
    
    # 3) compute delta of MFCC[0] (first coefficient)
    delta_mfcc = librosa.feature.delta(mfcc)[0]      # shape=(n_frames,)
    delta_mean = np.mean(delta_mfcc)                 # single scalar

    # 4) compute spectral centroid per frame
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  
    centroid_mean = np.mean(centroids)  
    
    #intensity extraction
    intensity = sound.to_intensity() 
    duration = sound.get_total_duration() #get total duration
    
    intensity_values = intensity.values.T.flatten() #count number of syllables based on intensity
    threshold = np.percentile(intensity_values, 75)
    
    peaks = np.where(intensity_values > threshold, 1,0)
    syllable_count = np.sum(peaks)
    
    speaking_rate = syllable_count / duration
    
    #return all outputs
    return{

        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'jitter': jitter,
        'shimmer': shimmer,
        
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'f1_min': f1_min,
        'f1_max': f1_max,
        'f1_range': f1_range,
        
        'f2_mean': f2_mean,
        'f2_std': f2_std,
        'f2_min': f2_min,
        'f2_max': f2_max,
        'f2_range': f2_range,
        
        'f3_mean': f3_mean,
        'f3_std': f3_std,
        'f3_min': f3_min,
        'f3_max': f3_max,
        'f3_range': f3_range,
        
        'f2_f1_ratio_mean': f2_f1_ratio_mean,
        
        'mfcc_1_mean': mfcc1_mean,
        'hnr_mean': hnr_mean,
        'hnr_std': hnr_std,
        'delta_mean': delta_mean,
        'centroid_mean': centroid_mean,

        'speaking_rate': speaking_rate
    }
    
