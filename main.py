"""
 Copyright (C) 2023 Fern Lane, WAV-Spectrum-Analyser project

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 See the License for the specific language governing permissions and
 limitations under the License.

 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
"""

import gc
import os.path
import wave

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__version__ = "1.0.0"

WAV_FILES_PATHS = [
    "samples/White Noise.wav",
    "samples/Pink Noise.wav",
    "samples/20Hz-20KHz Sweep.wav",
    "samples/100Hz Sine.wav",
    "samples/440Hz Sine.wav",
    "samples/1KHz Sine.wav",
    "samples/5KHz Sine.wav"
]

#  0  - int8 [b / (2^8 / 2)] (aka Signed 8-bit PCM)
#  1  - uint8 [(B - (2^8 / 2)) / (2^8 / 2)] (aka Unsigned 8-bit PCM)
# *2  - Little-endian int16 [<h / (2^16 / 2)] (aka Signed 16-bit PCM)
#  3  - Big-endian int16 [>h / (2^16 / 2)]
#  4  - Little-endian uint16 [(<H - (2^16 / 2)) / (2^16 / 2)] (aka Unsigned 16-bit PCM)
#  5  - Big-endian uint16 [(>H - (2^16 / 2)) / (2^16 / 2)]
# *6  - Little-endian int24 [<i / (2^24 / 2)] (aka Signed 24-bit PCM)
#  7  - Big-endian int24 [>i / (2^24 / 2)]
#  8 - Little-endian uint24 [(<I - (2^24 / 2)) / (2^24 / 2)] (aka Unsigned 24-bit PCM)
#  9  - Big-endian uint24 [(>I - (2^24 / 2)) / (2^24 / 2)]
#  10 - Little-endian int32 [<i / (2^32 / 2)] (aka Signed 32-bit PCM)
#  11 - Big-endian int32 [>i / (2^32 / 2)]
#  12 - Little-endian uint32 [(<I - (2^32 / 2)) / (2^32 / 2)] (aka Unsigned 32-bit PCM)
#  13 - Big-endian uint32 [(>I - (2^32 / 2)) / (2^32 / 2)]
#  14 - Little-endian float32 [<f] (aka 32-bit float)
#  15 - Big-endian float32 [>f]
WAV_FILE_FORMAT = 6

# 0 - No window (rectangular)
# 1 - Hamming window
# 2 - Hann window
# 3 - Blackman window
FFT_WINDOW = 0

# 0 - No normalization
# 1 - Averaging normalization
# 2 - Peak normalization
NORMALIZE_DATA = 2

# Minimum frequency (in Hz) to use in plotting and data extraction
MIN_FREQUENCY_HZ = 10

# Minimum FFT amplitude on the plot (in dBFS)
PLOT_MIN_POWER_DBFS = -100

# Number of points of each plot (higher values -> higher resolution, but the readability of the plot is lower)
POINTS_N = 1000

# -1 - Don't extract any data
# 0 - Extract average (mean) as calibration profile
# 1 to N - Extract calibration profile for specific file (in WAV_FILES_PATHS) referenced to average
# Set EXTRACT_DATA_REDUCED_POINTS to True to reduce number of points to POINTS_N
EXTRACT_DATA = -1
EXTRACT_DATA_PATH = "C:\\Users\\F3rni\\Desktop\\Calibration.txt"
EXTRACT_DATA_DELIMITER = "\t"
EXTRACT_DATA_REDUCED_POINTS = True


def read_wave_file(file_path: str) -> (np.ndarray, int):
    """
    Reads samples from file
    :param file_path: path to .wav file (use WAV_FILE_FORMAT to select format)
    :return: ([array of samples in float32 format (1st channel only)], sampling rate)
    """
    # Open file
    print("Reading {} file".format(file_path))
    wave_file = wave.open(file_path, "r")

    # Read frames
    frames = wave_file.readframes(wave_file.getnframes())

    # Get format
    if WAV_FILE_FORMAT == 0:
        print("Using format int8")
        audio_data = np.asarray(np.frombuffer(frames, dtype=np.int8), dtype=np.float32) / ((1 << 8) / 2.)

    elif WAV_FILE_FORMAT == 1:
        print("Using format uint8")
        audio_data = np.asarray(np.frombuffer(frames, dtype=np.uint8), dtype=np.float32) - ((1 << 8) / 2.)
        audio_data /= ((1 << 8) / 2.)

    elif WAV_FILE_FORMAT == 2:
        print("Using format Little-endian int16")
        dtype = np.dtype(np.int16)
        dtype = dtype.newbyteorder("<")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) / ((1 << 16) / 2.)

    elif WAV_FILE_FORMAT == 3:
        print("Using format Big-endian int16")
        dtype = np.dtype(np.int16)
        dtype = dtype.newbyteorder(">")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) / ((1 << 16) / 2.)

    elif WAV_FILE_FORMAT == 4:
        print("Using format Little-endian uint16")
        dtype = np.dtype(np.uint16)
        dtype = dtype.newbyteorder("<")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) - ((1 << 16) / 2.)
        audio_data /= ((1 << 16) / 2.)

    elif WAV_FILE_FORMAT == 5:
        print("Using format Big-endian uint16")
        dtype = np.dtype(np.uint16)
        dtype = dtype.newbyteorder(">")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) - ((1 << 16) / 2.)
        audio_data /= ((1 << 16) / 2.)

    elif WAV_FILE_FORMAT == 6:
        print("Using format Little-endian int24")
        # Add extra 0 after each value to simulate uint32 little-endian values
        frames_uint8 = np.frombuffer(frames, dtype=np.uint8)
        frames_uint8 = np.insert(frames_uint8, np.arange(3, len(frames_uint8) + 1, 3), 0)

        # Convert to uint32
        dtype = np.dtype(np.int32)
        dtype = dtype.newbyteorder("<")
        frames_uint32 = np.frombuffer(frames_uint8, dtype=dtype)

        # Convert negative values
        frames_uint32[frames_uint32 >= 0x800000] -= 1 << 24

        # Convert to float32
        audio_data = np.asarray(frames_uint32, dtype=np.float32) / ((1 << 24) / 2.)

    elif WAV_FILE_FORMAT == 7:
        print("Using format Big-endian int24")
        # Add extra 0 before each value to simulate uint32 big-endian values
        frames_uint8 = np.frombuffer(frames, dtype=np.uint8)
        frames_uint8 = np.insert(frames_uint8, np.arange(0, len(frames_uint8), 3), 0)

        # Convert to uint32
        dtype = np.dtype(np.int32)
        dtype = dtype.newbyteorder(">")
        frames_uint32 = np.frombuffer(frames_uint8, dtype=dtype)

        # Convert negative values
        frames_uint32[frames_uint32 >= 0x800000] -= 1 << 24

        # Convert to float32
        audio_data = np.asarray(frames_uint32, dtype=np.float32) / ((1 << 24) / 2.)

    elif WAV_FILE_FORMAT == 8:
        print("Using format Little-endian uint24")
        # Add extra 0 after each value to simulate uint32 little-endian values
        frames_uint8 = np.frombuffer(frames, dtype=np.uint8)
        frames_uint8 = np.insert(frames_uint8, np.arange(3, len(frames_uint8) + 1, 3), 0)

        # Convert to uint32
        dtype = np.dtype(np.uint32)
        dtype = dtype.newbyteorder("<")
        frames_uint32 = np.frombuffer(frames_uint8, dtype=dtype)

        # Convert to float32
        audio_data = np.asarray(frames_uint32, dtype=np.float32) - ((1 << 32) / 2.)
        audio_data /= ((2 << 23) / 2.)

    elif WAV_FILE_FORMAT == 9:
        print("Using format Big-endian uint24")
        # Add extra 0 before each value to simulate uint32 big-endian values
        frames_uint8 = np.frombuffer(frames, dtype=np.uint8)
        frames_uint8 = np.insert(frames_uint8, np.arange(0, len(frames_uint8), 3), 0)

        # Convert to uint32
        dtype = np.dtype(np.int32)
        dtype = dtype.newbyteorder(">")
        frames_uint32 = np.frombuffer(frames_uint8, dtype=dtype)

        # Convert to float32
        audio_data = np.asarray(frames_uint32, dtype=np.float32) - ((1 << 32) / 2.)
        audio_data /= ((2 << 23) / 2.)

    elif WAV_FILE_FORMAT == 10:
        print("Using format Little-endian int32")
        dtype = np.dtype(np.int32)
        dtype = dtype.newbyteorder("<")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) / ((1 << 32) / 2.)

    elif WAV_FILE_FORMAT == 11:
        print("Using format Big-endian int32")
        dtype = np.dtype(np.int32)
        dtype = dtype.newbyteorder(">")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) / ((1 << 32) / 2.)

    elif WAV_FILE_FORMAT == 12:
        print("Using format Little-endian uint32")
        dtype = np.dtype(np.uint32)
        dtype = dtype.newbyteorder("<")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) - ((1 << 32) / 2.)
        audio_data /= ((2 << 31) / 2.)

    elif WAV_FILE_FORMAT == 13:
        print("Using format Big-endian uint32")
        dtype = np.dtype(np.uint32)
        dtype = dtype.newbyteorder(">")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32) - ((1 << 32) / 2.)
        audio_data /= ((1 << 32) / 2.)

    elif WAV_FILE_FORMAT == 14:
        print("Using format Little-endian float32")
        dtype = np.dtype(np.float32)
        dtype = dtype.newbyteorder("<")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32)

    elif WAV_FILE_FORMAT == 15:
        print("Using format Big-endian float32")
        dtype = np.dtype(np.float32)
        dtype = dtype.newbyteorder(">")
        audio_data = np.asarray(np.frombuffer(frames, dtype=dtype), dtype=np.float32)

    else:
        raise ValueError("Invalid WAV_FILE_FORMAT")

    # Use only first channel
    samples = audio_data[::wave_file.getnchannels()]

    # Clear possible garbage
    gc.collect()

    # Print debug info
    print("Decoded {} samples (~{:.2f}s) @ {}Hz."
          " Min value: {:.2f}, max value: {:.2f}".format(len(samples),
                                                         len(samples) / wave_file.getframerate(),
                                                         wave_file.getframerate(),
                                                         np.min(samples),
                                                         np.max(samples)))

    # Return decoded samples and sampling rate
    return samples, wave_file.getframerate()


def log_decimation(input_data: np.ndarray, num_points: int, index_start=0, filter_before=True):
    """
    Logarithmically reduces the number of points in data
    :param input_data: data array to reduce number of points
    :param num_points: number of points in the reduced output
    :param index_start: initial (first) index (starting from 0)
    :param filter_before: set to True to apply sinc filter before decimating
    :return: reduced data
    """
    # Check number of points
    if num_points >= len(input_data):
        return input_data[index_start:]

    # Apply sinc filter before decimating
    # Filter code: https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
    if filter_before:
        # Define the cutoff frequency and the transition bandwidth
        fc = 0.01  # Cutoff frequency as a fraction of the sampling rate
        b = 0.08  # Transition band, as a fraction of the sampling rate

        # Calculate the filter length, must be odd
        filter_length = int(np.ceil((4 / b)))
        if not filter_length % 2:
            filter_length += 1  # Make sure that filter_length is odd

        # Create an array of filter coefficients
        n = np.arange(filter_length)

        # Compute sinc filter
        h = np.sinc(2 * fc * (n - (filter_length - 1) / 2))

        # Compute Blackman window
        # window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (filter_length - 1)) \
        #          + 0.08 * np.cos(4 * np.pi * n / (filter_length - 1))
        window = np.blackman(filter_length)

        # Multiply sinc filter by window
        h = h * window

        # Normalize to get unity gain at DC
        h = h / np.sum(h)

        # Apply sinc filter to the input signal
        input_data = np.convolve(input_data, h, mode="same")

    # Calculate the logarithmic spacing of indices
    indices = np.logspace(np.log10(index_start + 1),
                          np.log10(len(input_data)),
                          num_points,
                          endpoint=True,
                          dtype=int) - 1

    # Uncomment to disable decimation
    # return input_data[index_start:]

    # Uncomment to use simple sampling points reducing
    # reduced_data = input_data[indices]
    # return reduced_data

    # Calculate the reduced FFT data (decimate)
    reduced_data = []
    for i in range(len(indices)):
        if i < len(indices) - 1:
            start = indices[i]
            stop = indices[i + 1]
            if stop > start:
                value_max = np.max(input_data[start: stop])
                value_min = np.min(input_data[start: stop])
                reduced_sample = (value_max + value_min) / 2
                # reduced_sample = np.mean(input_data[start: stop])
                # reduced_sample = input_data[start]
                reduced_data.append(reduced_sample)
            else:
                reduced_data.append(input_data[start])

    return np.asarray(reduced_data, dtype=np.float32)


def main() -> None:
    """
    Main program entry point
    :return:
    """
    # Print version
    print("WAV-Spectrum-Analyser v.{}".format(__version__))

    # Read all files
    samples_all = []
    sampling_rate_prev = -1
    samples_len_min = np.inf
    for i in range(len(WAV_FILES_PATHS)):
        samples, sampling_rate = read_wave_file(WAV_FILES_PATHS[i])

        # Check sampling rate
        if sampling_rate_prev < 0:
            sampling_rate_prev = sampling_rate
        if sampling_rate != sampling_rate_prev:
            raise Exception("Sampling rates are not equal!")

        # Get minimum length
        if len(samples) < samples_len_min:
            samples_len_min = len(samples)

        # Append samples
        samples_all.append(samples)

    # Clear possible garbage
    gc.collect()

    ffts_dbfs = []
    ffts_dbfs_decimated = []

    # Calculate fft window
    if FFT_WINDOW == 0:
        fft_window = np.ones(samples_len_min, dtype=np.float32)
    elif FFT_WINDOW == 1:
        fft_window = np.hamming(samples_len_min)
    elif FFT_WINDOW == 2:
        fft_window = np.hanning(samples_len_min)
    elif FFT_WINDOW == 2:
        fft_window = np.blackman(samples_len_min)
    else:
        raise ValueError("Invalid FFT_WINDOW")

    # Calculate fft frequencies
    fft_freqs = np.fft.fftfreq(samples_len_min, d=1. / sampling_rate_prev)
    fft_freqs = fft_freqs[:len(fft_freqs) // 2]

    # Calculate start index
    index_start = int(MIN_FREQUENCY_HZ * len(fft_freqs) / (sampling_rate_prev / 2))

    # Reduce number of data point (for plotting)
    fft_freqs_decimated = log_decimation(fft_freqs, POINTS_N, index_start=index_start, filter_before=False)

    # Initialize matplotlib
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    fig.tight_layout(pad=3)

    # Initialize plot
    axs.set_title("Spectrum power" + (" (normalized)" if NORMALIZE_DATA > 0 else ""))
    axs.grid(True, which="both")
    axs.set(xlabel="Frequency (Hz)", ylabel="Amplitude (dBFS)")
    axs.set_xscale("log", base=10)
    axs.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Process all samples from buffer
    for i in range(len(WAV_FILES_PATHS)):
        # Cut to minimum size
        extra_size = (len(samples_all[i]) - samples_len_min) // 2
        if extra_size > 0:
            samples_all[i] = samples_all[i][extra_size:-extra_size]

        # Calculate FFT
        fft = np.fft.rfft(samples_all[i])

        # Scale the magnitude of FFT
        # fft_mag = np.abs(fft_real) * 2 / (len(samples_all[i]) / 2)
        fft_mag = np.abs(fft) * 2 / np.sum(fft_window)
        fft_mag = np.real(fft_mag[:-1])

        # Reduce number of points to plot
        fft_mag_decimated = log_decimation(fft_mag, POINTS_N, index_start=index_start)

        # Convert to dBFS
        min_value = np.finfo(np.float32).eps
        fft_mag[fft_mag < min_value] = min_value
        fft_mag_decimated[fft_mag_decimated < min_value] = min_value
        fft_dbfs = 20 * np.log10(fft_mag)
        fft_dbfs_decimated = 20 * np.log10(fft_mag_decimated)

        # Print raw info
        print("{}, before normalization: Average: {:.2f} dBFS,"
              " Peak: {:.2f} dBFS @ {:.2f} Hz".format(os.path.basename(WAV_FILES_PATHS[i]),
                                                      np.mean(fft_dbfs), np.max(fft_dbfs),
                                                      fft_freqs[np.where(fft_dbfs == np.max(fft_dbfs))[0][0]]))

        # Normalize FFTs
        if NORMALIZE_DATA > 0:
            if NORMALIZE_DATA == 1:
                print("Using averaging normalization")
                fft_dbfs = fft_dbfs - np.mean(fft_dbfs)
                fft_dbfs_decimated = fft_dbfs_decimated - np.mean(fft_dbfs_decimated)
            elif NORMALIZE_DATA == 2:
                print("Using peak normalization")
                fft_dbfs = fft_dbfs - np.max(fft_dbfs)
                fft_dbfs_decimated = fft_dbfs_decimated - np.max(fft_dbfs_decimated)
            else:
                raise ValueError("Invalid NORMALIZE_DATA")

            # Print normalized data info
            print("{}, after normalization: Average: {:.2f} dBFS,"
                  " Peak: {:.2f} dBFS @ {:.2f} Hz".format(os.path.basename(WAV_FILES_PATHS[i]),
                                                          np.mean(fft_dbfs), np.max(fft_dbfs),
                                                          fft_freqs[np.where(fft_dbfs == np.max(fft_dbfs))[0][0]]))

        # Append calculated FFT to list
        ffts_dbfs.append(fft_dbfs)
        ffts_dbfs_decimated.append(fft_dbfs_decimated)

        # Plot data
        fft_dbfs_plot = fft_dbfs_decimated
        fft_dbfs_plot[fft_dbfs_plot < PLOT_MIN_POWER_DBFS] = PLOT_MIN_POWER_DBFS
        axs.plot(fft_freqs_decimated, fft_dbfs_plot, label=os.path.basename(WAV_FILES_PATHS[i]), alpha=0.5)

    # Convert to numpy arrays
    ffts_dbfs = np.asarray(ffts_dbfs, dtype=np.float32)
    ffts_dbfs_decimated = np.asarray(ffts_dbfs_decimated, dtype=np.float32)

    # Calculate average
    fft_mean = ffts_dbfs.mean(axis=0)
    fft_mean_decimated = ffts_dbfs_decimated.mean(axis=0)

    ffts_dbfs_extract = ffts_dbfs_decimated if EXTRACT_DATA_REDUCED_POINTS else ffts_dbfs[index_start:]
    fft_freqs_extract = fft_freqs_decimated if EXTRACT_DATA_REDUCED_POINTS else fft_freqs[index_start:]
    fft_mean_extract = fft_mean_decimated if EXTRACT_DATA_REDUCED_POINTS else fft_mean[index_start:]

    # Extract data
    if EXTRACT_DATA >= 0:
        # Header
        calibration_lines = [["Freq [Hz]", "dBRel"]]

        # Mean calibration
        if EXTRACT_DATA == 0:
            print("Extracting average (mean) data as calibration profile")
            for i in range(len(fft_mean_extract)):
                calibration_lines.append(["{:.2f}".format(fft_freqs_extract[i]),
                                          "{:.2f}".format(-fft_mean_extract[i])])

        # Relative calibration
        elif 0 < EXTRACT_DATA <= len(WAV_FILES_PATHS):
            print("Extracting {} file data vs average as calibration profile"
                  .format(os.path.basename(WAV_FILES_PATHS[EXTRACT_DATA - 1])))
            for i in range(len(fft_mean_extract)):
                calibration_lines.append(["{:.2f}".format(fft_freqs_extract[i]),
                                          "{:.2f}".format(fft_mean_extract[i]
                                                          - ffts_dbfs_extract[EXTRACT_DATA - 1][i])])

        # Write to file
        print("Saving extracted data as {}".format(EXTRACT_DATA_PATH))
        with open(EXTRACT_DATA_PATH, "w") as file:
            for line in calibration_lines:
                file.write(EXTRACT_DATA_DELIMITER.join(line))
                file.write("\n")

    # Plot average
    axs.plot(fft_freqs_decimated, fft_mean_decimated, label="Average")

    # Show plot
    fig.legend(loc="upper right", framealpha=0.5)
    fig.show()


if __name__ == "__main__":
    main()
