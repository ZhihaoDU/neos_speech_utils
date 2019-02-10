import numpy as np
import librosa
from my_logger import get_my_logger

# Author:   Zhihao Du
# Email:    duzhihao.china@gmail.com


def griffin_lim(speech_mag, win_len, hop_len, max_iteration, rtol=1e-4):
    """
    Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    :param speech_mag: The magnitude spectrogram to be used for reconstruction.
    A 2-D numpy array with FxT dimensions, the first dim F is frequency bins, the second dim T is frame numbers.
    :param win_len: The window length for FFT and iFFT. Integer.
    :param hop_len: The hop length for FFT and iFFT (also named window shift length). Integer.
    :param max_iteration: The maximum iteration of Griffin Lim algorithm. Integer.
    :param rtol: The relative MSE of pre_step and current step. This function will try to match this tolerance util
    reach the max_iteration. Float.
    :return:
        signal: The reconstructed signal the magnitude of which is given in the speech_mag. A 1-D numpy array.
        MSE: The reconstruction loss.
    """

    frequency_bin, frame_num = speech_mag.shape
    if frequency_bin != win_len // 2 + 1:
        raise Exception("The window length {:d} and the frequency bins {:d} of magnitude are not matched.".
                        format(win_len, frequency_bin))

    signal_len = (frame_num - 1) * hop_len + win_len
    signal = np.random.randn(signal_len)
    pre_mse = 0.
    for i in range(max_iteration):
        # stft = stft_for_reconstruction(signal, win_len, hop_len)
        X = librosa.stft(signal, win_len, hop_len, win_len, center=False)
        angle = np.angle(X)
        modified_stft = speech_mag * np.exp(1.0j * angle)
        x = librosa.istft(modified_stft, 160, 320, center=False, length=signal_len)
        current_mse = np.sqrt(np.mean(np.square(x - signal)))
        if i % 10 == 0:
            logger.info("Iteration {:04d}: MSE: {:.6f}".format(i, current_mse))
        # if abs(current_mse - pre_mse) < rtol:
        #     return istft, current_mse
        pre_mse = current_mse
        signal = x
    # Smooth the first-half and last-half parts in the reconstructed signal.
    ifft_window = librosa.filters.get_window('hann', win_len, True)
    ifft_window = librosa.util.pad_center(ifft_window, win_len)
    signal[:160] *= ifft_window[:160]
    signal[-160:] *= ifft_window[-160:]
    return signal, pre_mse


if __name__ == '__main__':
    # cute 10 seconds slice from foo.wav
    # speech, sr = librosa.load("wavs/foo.wav", 16000, duration=10)
    # librosa.output.write_wav("wavs/origin.wav", speech, 16000, norm=True)
    # exit(0)
    logger = get_my_logger("griffin_lim")
    speech, sr = librosa.load("wavs/origin.wav", 16000, duration=10)
    # speech_mag = np.abs(stft_for_reconstruction(speech, 320, 160))
    speech_mag = np.abs(librosa.stft(speech, 320, 160, 320, center=False))
    reconstructed_speech, e = griffin_lim(speech_mag, 320, 160, 1000, 1e-6)
    logger.info("MSE: %.6f" % e)
    librosa.output.write_wav("wavs/reconstructed_1k.wav", reconstructed_speech, 16000, norm=True)
