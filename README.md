# ECG and Heart Sound Analysis Project 🫀

## Overview 📊
This project implements a comprehensive analysis of ECG (Electrocardiogram) signals and heart sounds, providing tools for signal processing, QRS detection, and heart rate variability analysis across different physical states.

## Table of Contents 📑
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Prerequisites 🔧
- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Installation 💻
1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```bash
pip install numpy matplotlib scipy
```

## Project Structure 📁
The project consists of three main Python scripts:
- `part1.py`: ECG signal processing and heart rate analysis
- `part2.py`: QRS detection and signal quality analysis
- `part3.py`: Heart sound (S1/S2) detection and analysis

## Features ⭐
- **Signal Processing** 🔍
  - Bandpass filtering (0.5-100 Hz)
  - FFT analysis
  - Signal quality assessment

- **ECG Analysis** 💓
  - QRS complex detection
  - R-peak identification
  - Heart rate calculation
  - Statistical analysis across different physical states

- **Heart Sound Analysis** 🔊
  - S1/S2 heart sound detection
  - Temporal analysis of heart sounds
  - Pre/post activity comparison

## Usage 🚀
1. Place your ECG data files in the project directory:
   - `108-a.txt`: Multi-lead ECG data
   - `108-b.txt`: Single-lead ECG data
   - `108-c.txt`: Combined ECG and heart sound data

2. Run the analysis scripts:
```bash
python part1.py  # For basic ECG analysis
python part2.py  # For QRS detection
python part3.py  # For heart sound analysis
```

## Results 📈
The scripts generate various visualizations and metrics:
- ECG waveform plots
- R-peak detection visualization
- Heart rate variability analysis
- Heart sound detection plots
- Statistical comparisons between rest and activity states

## Contributing 🤝
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License 📄
[Your chosen license]
