import numpy as np
import wave
import time
from scipy.signal import upfirdn
import adi
import matplotlib.pyplot as plt

def rrc_filter(beta, sps, num_taps):
    """
    Generate Root Raised Cosine (RRC) filter taps based on the exact given mathematical expression.
    """
    T = 1  # Assume symbol period = 1
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps  # centered time vector

    h = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = (1.0 / T) * (1 + beta * (4/np.pi - 1))
        elif np.abs(t[i]) == T / (4*beta):
            h[i] = (beta / (T * np.sqrt(2))) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            numerator = np.sin(np.pi*t[i]*(1-beta)/T) + \
                        4*beta*t[i]*np.cos(np.pi*t[i]*(1+beta)/T)
            denominator = np.pi*t[i](1 - (4*beta*t[i]/T)*2)
            h[i] = (1/T) * (numerator / denominator)
    
    h = h / np.sqrt(np.sum(h**2))  # Normalize energy
    return h

def wav_to_binary(filename, num_bits=8):
    with wave.open(filename, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()

        frames = wav_file.readframes(n_frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        audio_data = np.frombuffer(frames, dtype=dtype)

        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(dtype)

        audio_data = audio_data.astype(np.float32)
        audio_min, audio_max = audio_data.min(), audio_data.max()
        if audio_max - audio_min > 0:
            audio_data = ((audio_data - audio_min) / (audio_max - audio_min) * 255).astype(np.uint8)

        bitstream = ''.join(format(byte, f'0{num_bits}b') for byte in audio_data)
        return bitstream

def qpsk_modulation(bitstream, preamble_bits=None):
    if preamble_bits is not None:
        bitstream = preamble_bits + bitstream

    bits = np.array([int(b) for b in bitstream])
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # pad if odd number

    odd_bits = bits[0::2]
    even_bits = bits[1::2]
    odd_bits = np.where(odd_bits == 0, -1, 1)
    even_bits = np.where(even_bits == 0, -1, 1)

    symbols = odd_bits + 1j * even_bits

    # Oversample
    os_factor = 4
    symbols_oversampled = upfirdn([1], symbols, up=os_factor)

    # Apply RRC filter
    num_taps = 101
    beta = 0.35
    h_rrc = rrc_filter(beta=beta, sps=os_factor, num_taps=num_taps)
    shaped_signal = np.convolve(symbols_oversampled, h_rrc, mode='same')

    # Modulate to passband
    fc = 2.4e9
    fs = 1e6
    t_passband = np.arange(len(shaped_signal)) / fs
    passband_signal = shaped_signal * np.exp(1j * 2 * np.pi * fc * t_passband)

    return passband_signal

def plot_psd(signal, fs):
    plt.figure(figsize=(10, 5))
    plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=True)
    plt.title("Power Spectral Density of Transmitted Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid()
    plt.show()

def plot_constellation(signal, os_factor=4):
    filter_delay = 50  # FIR filter delay
    signal_downsampled = signal[filter_delay::os_factor]

    plt.figure(figsize=(6, 6))
    plt.plot(np.real(signal_downsampled), np.imag(signal_downsampled), 'o', markersize=2, alpha=0.5, label='Received Symbols')

    ideal_points = np.array([
        [0, -1],
        [-1, 0],
        [0, 1],
        [1, 0]
    ])
    plt.plot(ideal_points[:, 0], ideal_points[:, 1], 'rx', markersize=10, label='Ideal QPSK Points')

    plt.grid(True)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("QPSK Constellation Diagram (Per Transmission)")
    plt.axis('equal')
    plt.legend()
    plt.show()

def extract_bits_from_signal(modulated_signal, os_factor=4):
    filter_delay = 50  # Approximate FIR filter delay
    signal_downsampled = modulated_signal[filter_delay::os_factor]

    I = np.real(signal_downsampled)
    Q = np.imag(signal_downsampled)

    bits = []
    for i, qpsk_symbol in enumerate(signal_downsampled):
        bits.append(1 if I[i] > 0 else 0)
        bits.append(1 if Q[i] > 0 else 0)

    return ''.join(map(str, bits))

def transmit_signal(signal, bitstream, fs=1e6, fc=2.4e9, os_factor=4):
    try:
        sdr = adi.Pluto("ip:192.168.2.1")
        sdr.sample_rate = int(fs)
        sdr.tx_lo = int(fc)
        sdr.tx_cyclic_buffer = False

        signal = signal / (np.max(np.abs(signal)) + 1e-6)  # Normalize
        tx_signal = (signal + 1j * np.zeros_like(signal)).astype(np.complex64)

        print("Transmission started...")

        iteration = 0
        while True:
            try:
                sdr.tx_destroy_buffer()
                sdr.tx(tx_signal)
                print(f"\nTransmission iteration {iteration + 1} completed")

                extracted_bits = extract_bits_from_signal(signal, os_factor)
                print("First 100 extracted bits:")
                print(extracted_bits[:100])

                max_amplitude = np.max(np.abs(tx_signal))
                print(f"Max amplitude: {max_amplitude:.4f}")

                plot_constellation(signal, os_factor=os_factor)

                iteration += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"Transmission failed: {e}")
                break

        print("Transmission stopped.")
    except KeyboardInterrupt:
        print("Transmission stopped by user.")
    except Exception as e:
        print(f"Error during transmission: {e}")
    finally:
        try:
            del sdr
        except NameError:
            pass

# === Main Execution ===
filename = "song3.wav"
barker_code = '1111100110101'  # Barker preamble

bitstream = wav_to_binary(filename)
passband_signal = qpsk_modulation(bitstream, preamble_bits=barker_code)
plot_psd(passband_signal, fs=1e6)
transmit_signal(passband_signal, bitstream, fs=1e6, fc=2.4e9, os_factor=4)
