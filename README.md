# auto-accompaniment-reboot

A real-time automatic accompaniment system for vocal performances.

## Prerequisites

### 1. Install Python Dependencies

It is highly recommended to use _conda_ for environment management, because _conda_ provides some packages that are difficult to install using other approaches. To use _conda_, you must have [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

Installing Python dependencies with _conda_ is fairly easy, just run the following command in the terminal:

```bash
conda env create -n auto-accompaniment-reboot -f environment.yml
```

### 2. Install FluidSynth Library

This project makes use of _FluidSynth_ as the real-time software synthesizer. Apart from _pyFluidSynth_, which is the Python binding for _FluidSynth_, you must also install _FluidSynth_.

For __Windows__ users,

1. Download 

## Compatibility

Tested on the following platforms:

- Python 3.7.13, _Windows 10 21H1 Build 19044_
- Python 3.8.13, _macOS 12.3.1 21E258 (Apple Silicon, with Rosetta)_
- Python 3.8.5, _macOS 12.2.1 21D62 (Intel)_

## Reference Projects

- https://github.com/rtchen/accompaniment
- https://github.com/MichaelZhangty/Auto-Accompaniment
- https://github.com/MichaelZhangty/Auto-Acco-2020
