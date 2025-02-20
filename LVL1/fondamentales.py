from hmac import digest_size

from encodings.idna import sace_prefix

from DataManager import DataManager
import wave
import librosa
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.spatial.distance import euclidean

class ReadAudioFile:
    def __init__(self, audio_path):
        self.audio_path = audio_path


    def read_wave(self):
        wave_data = wave.open(self.audio_path, 'r')
        sample_freq = wave_data.getframerate()
        n_channels = wave_data.getnchannels()
        n_sample = wave_data.getnframes()
        time_audio = n_sample / sample_freq
        signal_wave = wave_data.readframes(n_sample)

        return sample_freq, n_channels, n_sample, time_audio, signal_wave


    def bytes_to_array(self, signal_wave, n_channels):
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        if n_channels == 2:
            left_channel = signal_array[0::2]
            right_channel = signal_array[1::2]
            return left_channel, right_channel
        return signal_array


    def linear_normalization(self, signal:np.ndarray):
        min_value = np.min(signal)
        max_value = np.max(signal)
        print(f"Valeur min avant normalisation : {min_value}")
        print(f"Valeur max avant normalisation : {max_value}")
        normalized_signal = (2*(signal - min_value) / (max_value-min_value)) - 1

        return normalized_signal


    def read_librosa(self):
        signal_wave, signal_rate = librosa.load(self.audio_path)

        return signal_wave, signal_rate


    def play_audio(self):
        ipd.Audio(self.audio_path)


    def display_temporal_signal(self, library:str='librosa'):
        plt.figure(figsize=(15, 5))
        if library == 'librosa':
            signal_wave, signal_rate = self.read_librosa()
            time_signal = signal_wave.shape[0]/signal_rate
            time = np.linspace(0, time_signal, num=signal_rate)
            plt.plot(time, signal_wave)
            plt.title('Temporal signal with librosa')

        elif library == 'wave':
            sample_freq, n_channels, n_sample, time_audio, signal_wave = self.read_wave()
            time = np.linspace(0, time_audio, num=n_sample)
            signal = self.bytes_to_array(signal_wave, n_channels)
            normalized_signal = signal / signal.max()
            plt.plot(time, normalized_signal)
            plt.title('Temporal signal with wave')


        plt.xlabel('Time (s)')
        plt.ylabel('Signal Wave')
        plt.show()


    def display_spectre(self, signal, sample_freq, window=None):
        if window == None:
            spectre = fft(signal)
        else:
            spectre = fft(signal[window])

        spectre /= spectre.max()
        n = spectre.size
        freq_axis = np.linspace(0, sample_freq, num=n)
        plt.vlines(freq_axis, [0], spectre, 'r')
        plt.axis([0, sample_freq//2, 0, 1])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title('Fft spectre')
        plt.show()


    def spectrogram_wave(self, signal, sample_freq, t_audio, win_size=512):
        hop_size = win_size // 2
        offset = win_size - hop_size
        n_windows = signal.shape[0] // offset if signal.shape[0] % offset == 0 else (signal.shape[0] // offset) + 1
        padding = offset * n_windows - signal.shape[0]
        padded_signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)
        spectrogram = np.zeros((win_size//2, n_windows))
        for i in range(n_windows - 1):
            start = i * hop_size
            stop = start + win_size
            windowed_signal = padded_signal[start:stop] * np.hanning(win_size)
            spectrum = np.abs(fft(windowed_signal))
            spectrogram[:, i] = spectrum[:win_size//2]

        spectrogram = 10 * np.log10(spectrogram + 1e-10)

        n_windows = spectrogram.shape[1]
        frequency = fftfreq(win_size, d=1/sample_freq)[:win_size//2]
        time = np.linspace(0, t_audio, num=n_windows)

        plt.figure(figsize=(15, 5))
        plt.pcolormesh(time, frequency, spectrogram, cmap='inferno')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram from scratch')
        plt.show()

        return spectrogram


    def spectrogram_scipy(self, signal, sample_freq):
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        frequencies, times, Sxx = spectrogram(signal, fs=sample_freq)
        plt.figure(figsize=(15, 5))
        plt.pcolormesh(times, frequencies, 10*np.log10(Sxx), shading='auto', cmap='inferno')
        plt.colorbar(label='Amplitude (dB)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram with spicy')
        plt.show()

        return 10*np.log10(Sxx)

    def display_stft(selfself, signal, sample_freq):
        stft = librosa.stft(signal)
        spectrogram = np.abs(stft)
        stft_db = librosa.amplitude_to_db(spectrogram)

        plt.figure(figsize=(15, 5))
        librosa.display.specshow(stft_db, sr=sample_freq, x_axis='time', y_axis='log', cmap='inferno')
        plt.colorbar(label='Amplitude (dB)')
        plt.title('STFT')
        plt.show()

        return stft_db


    def mel_filter_bank(self, n_filters, win_size, sample_freq, fmin=0, fmax=None):
        if fmax is None:
            fmax = sample_freq/2

        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)

        mel_points = np.linspace(mel_min, mel_max, num=n_filters + 2)
        freqs = 700 * (10**(mel_points / 2595) - 1)
        filter_bank = np.zeros((n_filters, win_size // 2))

        for i in range(1, n_filters+1):
            f_m1 = int(freqs[i-1] * win_size / sample_freq)
            f_m2 = int(freqs[i] * win_size / sample_freq)
            f_m3 = int(freqs[i+1] * win_size / sample_freq)

            filter_bank[i - 1, f_m1:f_m2] = (np.arange(f_m1, f_m2) - f_m1) / (f_m2 - f_m1)
            filter_bank[i - 1, f_m2:f_m3] = (f_m3 - np.arange(f_m2, f_m3)) / (f_m3 - f_m2)

        enorm = 2.0 / (freqs[2:n_filters+2] - freqs[:n_filters])
        filter_bank_norm = filter_bank * enorm[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i in range(n_filters):
            axes[0].plot(filter_bank[i])
        axes[0].set_title("Unnormalized Filters")
        axes[0].set_xlabel("Frequency [Hz]")
        axes[0].set_ylabel("Amplitude")

        for i in range(n_filters):
            axes[1].plot(filter_bank_norm[i])
        axes[1].set_title("Normalized Filters")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Amplitude")

        plt.tight_layout()  # Ajuster la mise en page pour éviter les chevauchements
        plt.grid(True)
        plt.show()

        return filter_bank, filter_bank_norm


    def display_mel_spectogram(self, signal, sample_freq, win_size, n_filters=40, fmin=0, fmax=None):
        hop_size = win_size // 2
        offset = win_size - hop_size
        n_windows = signal.shape[0] // offset if signal.shape[0] % offset == 0 else (signal.shape[0] // offset) + 1
        padding = offset * n_windows - signal.shape[0]
        padded_signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)

        mel_spec = np.zeros((n_filters, n_windows))

        filter_bank, filter_bank_norm = self.mel_filter_bank(n_filters, win_size, sample_freq, fmin, fmax)

        for i in range(n_windows - 1):
            start = i * hop_size
            stop = start + win_size
            windowed_signal = padded_signal[start:stop] * np.hanning(win_size)

            spectrum = np.abs(fft(windowed_signal))[:win_size // 2]

            mel_spec[:, i] = np.dot(filter_bank_norm, spectrum)

        mel_spec = 10 * np.log10(mel_spec + 1e-10)

        n_windows = mel_spec.shape[1]
        frequency = np.linspace(0, sample_freq // 2, num=mel_spec.shape[0])
        time = np.linspace(0, signal.size / sample_freq, num=n_windows)

        plt.figure(figsize=(15, 5))
        plt.pcolormesh(time, frequency, mel_spec, shading='auto', cmap='inferno')
        plt.colorbar(label='Amplitude (dB)')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Mel Spectrogram')
        # plt.yscale('log')
        plt.show()

        return mel_spec


    def display_mel_spectrogram_librosa(self, signal, sample_freq, n_filters=40):
        S = librosa.feature.melspectrogram(y=signal, sr=sample_freq, n_mels=n_filters)

        log_S = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(15, 5))
        librosa.display.specshow(log_S, sr=sample_freq, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()


    def display_MFCCs(self, signal, sample_freq, dct_filter_num=40):
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_freq, n_mels=dct_filter_num)
        filter_len = mel_spec.shape[0]
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)
        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        cepstral_coefficients = np.dot(basis, mel_spec)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        time_axis = np.linspace(0, signal.size / sample_freq, num=len(signal))
        axes[0].plot(time_axis, signal)
        axes[0].set_title("Audio Signal")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")

        img = axes[1].imshow(cepstral_coefficients, aspect='auto', origin='lower', cmap='inferno')
        axes[1].set_title("Mel-Frequency Cepstral Coefficients (MFCCs)")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("MFCC Index")

        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()

        return cepstral_coefficients


    def chroma_features(self, signal, sample_freq):
        chroma = librosa.feature.chroma_stft(y=signal, sr=sample_freq)

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
        plt.colorbar(label='Amplitude')
        plt.title('Chroma Features')
        plt.show()

        return chroma

    def extract_voiceprint(self, mfcc):
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        voiceprint = np.concatenate([mfcc_mean, mfcc_std])

        return voiceprint


    def compare_voiceprints(self, mfcc_voice1, mfcc_voice2):
        voiceprint1 = self.extract_voiceprint(mfcc_voice1)
        voiceprint2 = self.extract_voiceprint(mfcc_voice2)

        distance = euclidean(voiceprint1, voiceprint2)
        print("Distance entre les voiceprints :", distance)

        return distance






