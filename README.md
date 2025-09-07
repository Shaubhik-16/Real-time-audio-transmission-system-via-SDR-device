# 📡 QPSK Transmitter & Receiver with Barker Code Synchronization (Pluto SDR)

This project implements a complete **QPSK communication system** using **Pluto SDR**. It includes a **transmitter** that prepends a **Barker code** to each frame and a **receiver** that detects this Barker sequence for **frame synchronization** and performs full demodulation and symbol recovery.

---

##  Features

### Transmitter:
-  **Barker Code Preamble** (13-bit) for sync detection
-  **QPSK Modulation** of binary payload
-  **Upsampling and Pulse Shaping** (e.g., RRC filter)
-  **Transmission** via Pluto SDR

### Receiver:
-  **Signal Acquisition** from Pluto SDR  
-  **Barker Code Detection** for frame alignment  
-  **Coarse Frequency Sync** (4th-power FFT method)  
-  **Mueller & Muller Clock Recovery** (Timing sync)  
-  **4th-Order Costas Loop** (Phase/frequency correction)  
-  **Constellation Diagrams** at each processing stage  
-  Optional saving of constellation figures

---

##  Setup

```bash
pip install numpy matplotlib scipy pyadi-iio
```

Ensure Pluto SDR is connected at `192.168.2.1`. You will need two Pluto SDRs for full duplex testing (one for TX, one for RX).

---

##  How to Run

### Transmitter

```bash
python sdr_transmitter_with_barker.py
```

- Continuously transmits QPSK symbols with a Barker preamble
- Add your message/payload in the script

### Receiver

```bash
python sdr_receiver_with_barker.py
```

- Detects Barker code and processes valid frames
- Synchronizes, demodulates, and plots constellations

---

##  Output (Receiver)

- **Terminal log**: Power level, synchronization status
- **Plots**: Constellation diagrams at key stages
- **Saved Figures**: `received_constellations/` directory (if enabled)

---

##  References

-  [pysdr.org](https://pysdr.org)  
-  Pluto SDR Docs: [Analog Devices Wiki](https://wiki.analog.com/university/tools/pluto)

---

##  Notes

- Barker code used: `[+1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 +1]` (13-bit)
- Match sampling rates (`fs`), symbol rates (`sps`), and Barker sequences in both TX & RX
