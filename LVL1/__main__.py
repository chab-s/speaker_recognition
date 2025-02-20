from fondamentales import ReadAudioFile
from DataManager import DataManager
from pydub import AudioSegment
import os

def count_wav_files(folder_path):
    return sum(1 for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(".wav"))

def combine_audios(folder_path, n_files:tuple, output_file=None):
    n_files_max = count_wav_files(folder_path)
    if len(n_files) > n_files_max:
        raise ValueError("le nombre de fichiers audio est insuffisant.")
    audio_files = [os.path.join(folder_path, f"{i}.wav") for i in range(*n_files) if
                   os.path.exists(os.path.join(folder_path, f"{i}.wav"))]
    if not audio_files:
        raise ValueError("La liste des fichiers audio est vide !")

    combined_audio = AudioSegment.from_file(audio_files[0])

    for file in audio_files[1:]:
        audio_segment = AudioSegment.from_file(file)
        combined_audio += audio_segment

    if output_file is not None:
        combined_audio.export(output_file, format="wav")
        print(f"✅ Fusion terminée : fichier '{output_file}' généré avec succès.")

    return combined_audio

def lvl1():
    dataset = 'kongaevans/speaker-recognition-dataset'
    data = DataManager(dataset)
    path = data.import_dataset()
    print(f'path to the dataset: {path}')
    folder_path1 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Magaret_Tarcher'
    folder_path2 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Nelson_Mandela'

    # reader = ReadAudioFile(audio)
    #
    # # play audio
    # print("Play audio")
    # reader.play_audio()
    #
    # # read audio
    # sample_freq, n_channels, n_sample, time_audio, signal_wave = reader.read_wave()
    # signal_wave_array = reader.bytes_to_array(signal_wave, n_channels)
    # signal_librosa, signal_rate_librosa = reader.read_librosa()
    # time = signal_librosa.size / signal_rate_librosa
    #
    # # Test d'affichage des résultats
    # print("Informations sur le fichier wave :")
    # print(f"Fréquence d'échantillonnage : {sample_freq} Hz")
    # print(f"Nombre de canaux : {n_channels}")
    # print(f"Nombre d'échantillons : {n_sample}")
    # print(f"Durée de l'audio : {time_audio} secondes")
    # print(f"Signal de l'onde (Wave) : {signal_wave_array}")
    # normalized_signal_wave = signal_wave_array / signal_wave_array.max()
    # print(f"Signal de l'onde normalise (Wave) : {normalized_signal_wave}")
    #
    # print("\nInformations sur le fichier Librosa :")
    # print(f"Signal Librosa : {signal_librosa}")
    # print(f"Taux d'échantillonnage Librosa : {signal_rate_librosa} Hz")

    # # display temporal signal
    # reader.display_temporal_signal('wave')
    # reader.display_temporal_signal('librosa')
    #
    # # display spectre fft
    # reader.display_spectre(signal_librosa, signal_rate_librosa)
    # reader.display_spectre(normalized_signal_wave, n_sample)
    #
    # # display spectrogram
    # reader.spectrogram_wave(signal_librosa, signal_rate_librosa, time, win_size=512)
    # reader.spectrogram_scipy(signal_librosa, signal_rate_librosa)
    #
    # # display the STFT
    # reader.display_stft(signal_librosa, signal_rate_librosa)
    #
    # # display MEL spectrogram
    # filter_bank, filter_bank_norm = reader.mel_filter_bank(40, 512, 16000)
    #
    # # display MEL spectrogram
    # reader.display_mel_spectogram(signal_librosa, signal_rate_librosa, win_size=512)
    # reader.display_mel_spectrogram_librosa(signal_librosa, signal_rate_librosa)
    # cepstral_coefficients = reader.display_MFCCs(signal_librosa, signal_rate_librosa)
    #
    # # display chroma features
    # reader.chroma_features(signal_librosa, signal_rate_librosa)

    # Compare voiceprints
    audio_path_MT1 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Magaret_Tarcher/175.wav'
    audio_path_MT2 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Magaret_Tarcher/177.wav'
    audio_path_NM1 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Nelson_Mandela/273.wav'
    audio_path_NM2 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/Nelson_Mandela/275.wav'

    audio_path1 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/tests/MT.wav'
    audio_path2 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/tests/MT2.wav'
    audio_path3 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/tests/NM.wav'
    audio_path4 = '/Users/mc/.cache/kagglehub/datasets/kongaevans/speaker-recognition-dataset/versions/1/16000_pcm_speeches/tests/NM2.wav'
    audio1 = combine_audios(folder_path1, (10, 20), output_file=audio_path1)
    audio2 = combine_audios(folder_path1, (0, 10), output_file=audio_path2)
    audio3 = combine_audios(folder_path2, (10, 20), output_file=audio_path3)
    audio4 = combine_audios(folder_path2, (20, 30), output_file=audio_path4)
    reader1 = ReadAudioFile(audio_path1)
    reader2 = ReadAudioFile(audio_path2)
    reader3 = ReadAudioFile(audio_path3)
    reader4 = ReadAudioFile(audio_path4)
    signal_librosa, signal_rate_librosa = reader1.read_librosa()
    signal_librosa2, signal_rate_librosa2 = reader2.read_librosa()
    signal_librosa3, signal_rate_librosa3 = reader3.read_librosa()
    signal_librosa4, signal_rate_librosa4 = reader4.read_librosa()
    cepstral_coefficients1 = reader1.display_MFCCs(signal_librosa, signal_rate_librosa)
    cepstral_coefficients2 = reader2.display_MFCCs(signal_librosa2, signal_rate_librosa2)
    cepstral_coefficients3 = reader1.display_MFCCs(signal_librosa3, signal_rate_librosa3)
    cepstral_coefficients4 = reader2.display_MFCCs(signal_librosa4, signal_rate_librosa4)
    distance_MT = reader1.compare_voiceprints(cepstral_coefficients1, cepstral_coefficients2)
    distance_NM = reader1.compare_voiceprints(cepstral_coefficients3, cepstral_coefficients4)
    distance_MT_NM1 = reader1.compare_voiceprints(cepstral_coefficients1, cepstral_coefficients3)
    distance_MT_NM2 = reader1.compare_voiceprints(cepstral_coefficients1, cepstral_coefficients4)
    distance_MT_NM3 = reader1.compare_voiceprints(cepstral_coefficients2, cepstral_coefficients3)
    distance_MT_NM4 = reader1.compare_voiceprints(cepstral_coefficients2, cepstral_coefficients4)

def lvl2():
    pass

if __name__ == '__main__':
    # lvl1()
    lvl2()
