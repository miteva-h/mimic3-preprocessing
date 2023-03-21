!pip install wfdb
!pip install detecta
!pip install pyocclient

import wfdb
import os
import numpy as np
import pandas as pd
from detecta import detect_peaks
from scipy import signal
import math
import owncloud
import pickle

#windows
w_flat = 15;    # flat lines window
w_peaks = 5;    # flat peaks window
w_fix = 15;     # flat join window

# thresholds
t_peaks = 0.2; # percentage of tolerated flat peaks
t_flat = 0.2;  # percentage of tolerated flat lines

#sampling rate
fsppg = 125


def flat_lines(data=None, window=None):
    # Flat line in ABP and PPG -> sliding window over the whole thing
    len = np.size(data)
    flat_locs_ppg = np.ones((len - window), dtype=int)

    # get the locations where i == i+1 == i+2 ... == i+window
    # efficient-ish sliding window
    for i in range(2, window):
        tmp_ppg = data[0:(len - window)] == data[i:(len - window + i)]
        flat_locs_ppg = np.logical_and(flat_locs_ppg, tmp_ppg)
        # print(flat_locs_ppg)

    # extend to be the same size as data
    flat_locs_ppg = np.concatenate([flat_locs_ppg, np.zeros(window)])
    # print(flat_locs_ppg)

    flat_locs_ppg2 = flat_locs_ppg

    for i in range(2, window):
        flat_locs_ppg[i:] = np.logical_or(flat_locs_ppg[i:], flat_locs_ppg2[1:(np.size(flat_locs_ppg2) - i + 1)])

    # print(flat_locs_ppg)
    per_ppg = np.sum(flat_locs_ppg) / len
    return per_ppg


def flat_peaks(data=None, window=None):
    # ppg_peaks   ... peak locations for PPG
    ppg_peaks = detect_peaks(data[0:], show=False)
    # ppg_valleys ... cycle start/end points for PPG
    ppg_valleys = detect_peaks(-1 * data[0:], show=False)
    # print(ppg_peaks)
    # print(ppg_valleys)

    number_of_peaks_ppg = np.size(ppg_peaks)
    number_of_valleys_ppg = np.size(ppg_valleys)
    # print(number_of_peaks_ppg)
    # print(number_of_valleys_ppg)

    # first get the flat lines:
    len = np.size(data)
    flat_locs_ppg = np.ones((len - window), dtype=int)

    # get the locations where i == i+1 == i+2 ... == i+window
    # efficient-ish sliding window
    for i in range(2, window):
        tmp_ppg = data[0:(len - window)] == data[i:(len - window + i)]
        flat_locs_ppg = np.logical_and(flat_locs_ppg, tmp_ppg)

    # extend to be the same size as data
    flat_locs_ppg = np.concatenate([flat_locs_ppg, np.zeros(window)])

    ppg_peak_ones = np.zeros(np.size(data[0:]))
    ppg_peak_ones[ppg_peaks] = 1;
    ppg_valley_ones = np.zeros(np.size(data[0:]))
    ppg_valley_ones[ppg_valleys] = 1;

    # extract the needed info:
    locs_of_flat_peaks_ppg = np.flatnonzero(np.logical_and(flat_locs_ppg, ppg_peak_ones));
    locs_of_flat_valleys_ppg = np.flatnonzero(np.logical_and(flat_locs_ppg, ppg_valley_ones));
    number_of_flat_peaks_ppg = np.size(locs_of_flat_peaks_ppg);
    number_of_flat_valleys_ppg = np.size(locs_of_flat_valleys_ppg);

    # print(locs_of_flat_peaks_ppg)
    # print(number_of_flat_valleys_ppg)

    skip_ppg = 0;

    if (number_of_flat_peaks_ppg >= t_peaks * number_of_peaks_ppg) or (
            number_of_flat_valleys_ppg >= t_peaks * number_of_valleys_ppg):
        skip_ppg = 1

    return skip_ppg


def filter_signal(data=None):
    ##  filter_signal
    # Remove steep steps at the beginning of the data (if there are any)
    dff = np.argwhere(np.abs(np.diff(data[0:100])) > 10)
    # dff[0] is the first such index
    if (np.size(dff) > 0):
        data = data[dff[0] + 1:]

    # In rare cases where the spike appears at the end of signal
    dff = np.argwhere(np.abs(np.diff(data[np.size(data) - 1 - 100:np.size(data) - 1])) > 10)
    # dff[0] is the first such index
    if (np.size(dff) > 0):
        data = data[0:np.size(data) - dff[0] - 1]
    return data;

def main():
    start_index = 0 # Change this value for each iteration by +1
    batch_size = 25
    patients_counter = 0
    np.set_printoptions(threshold=np.inf)
    one_second_list = []
    one_minute_list = []
    one_second_file = []
    one_minute_file = []

    # Reading all record names of the database and sorting them
    list = wfdb.get_record_list('mimic3wdb')
    list.sort()

    # Iterating through the record names list
    for index_i, i in enumerate(list[start_index * batch_size:(start_index + 1) * batch_size]):
        print("////////////////////////////////////////////////////////")
        print(index_i)
        print(i)
        try:
            # Reading the corresponding physiologic waveform record header
            record = wfdb.rdheader(i.split('/')[1], pn_dir='mimic3wdb/' + i)
            # Reading the corresponding numerics record header
            recordN = wfdb.rdheader(i.split('/')[1] + 'n', pn_dir='mimic3wdb/' + i)
            # Reading the corresponding layout header
            recordLayout = wfdb.rdheader(i.split('/')[1] + '_layout', pn_dir='mimic3wdb/' + i)
        except:
            print("Not found")
            continue

        # Printing the signal names in both records
        names = recordLayout.sig_name
        print(names)
        namesN = recordN.sig_name
        print(namesN)

        # Taking the PLETH (PPG signal) header from the physiologic waveform record
        ppg_header = [names.index(j) for j in names if 'PLETH' in j]
        # Taking the SpO2 header from the numerics record
        spo2_header = [namesN.index(j) for j in namesN if 'SpO2' in j]

        # Checking whether both headers are present
        if len(ppg_header) > 0 and len(spo2_header) > 0:
            print("There is a PLETH signal and SpO2 information")

            ppg_signal_length = record.sig_len
            spo2_numerics_length = recordN.sig_len

            # Printing the lengths of the corresponding PPG and SpO2 signals
            print("Length of PPG signal: " + str(ppg_signal_length))
            print("Length of SpO2 signal: " + str(spo2_numerics_length))

            # Calculating and printing the average PPG length per 1 SpO2 value
            length = math.floor(ppg_signal_length / spo2_numerics_length)
            print("Length of each PPG signal segment: " + str(length))

            # Filter #1: Discard all records where each SpO2 value is NOT linked to a PPG segment of duration 1 second or 1 minute
            if length in (fsppg, fsppg * 60):
                # print("Filter #1 OK! Each PPG segment is 1 second or 1 minute in length.")

                # Taking the values by reading the corresponding sample
                ppg_signal, ppg_fields = wfdb.rdsamp(i.split('/')[1], pn_dir='mimic3wdb/' + i, channel_names=[str(names[ppg_header[0]])])
                spo2_param, sp02_fields = wfdb.rdsamp(i.split('/')[1] + 'n', pn_dir='mimic3wdb/' + i, channel_names=[str(namesN[spo2_header[0]])])

                print("Length of downloaded PPG signal: " + str(len(ppg_signal)))
                print("Length of downloaded SpO2 signal: " + str(len(spo2_param)))
                counter = 0
                percentage_saved = 0

                # Normalization of the whole PPG record
                ppg_signal = (ppg_signal - np.nanmin(ppg_signal)) / (np.nanmax(ppg_signal) - np.nanmin(ppg_signal))

                # Iterating through all SpO2 values and constructing a corresponding dictionary object
                for spo2 in spo2_param:
                    # Filter #2: Check if the corresponding SpO2 value is not [nan]
                    if not math.isnan(spo2):
                        # print("Filter #2 OK! The corresponding SpO2 value is not [nan].")
                        # Taking the corresponding PPG segment
                        ppg_signal_part = ppg_signal[length * counter:length * (counter + 1)]

                        # Filter #3: Detecting flat lines in the signal
                        data = ppg_signal_part.reshape(np.size(ppg_signal_part), )
                        p_ppg = flat_lines(data, w_flat)
                        if (p_ppg < t_flat):
                            # print("Filter #3 OK! The PPG segment passed the flat lines threshold.")
                            # Filter #4: Detecting flat peaks in the signal
                            skip_ppg = flat_peaks(data, w_peaks)
                            if skip_ppg == 0:
                                # print("Filter #4 OK! The PPG segment passed the flat peaks threshold.")

                                # Filtering the PPG segment - removing noise at the beginning and at the end of the segment
                                data = filter_signal(data)
                                ppg_signal_part = data

                                # Building and saving the dictionary object
                                dictionary = {
                                    "ID": i,
                                    "PPG": ppg_signal_part,
                                    "SpO2": spo2[0],
                                }
                                percentage_saved += 1
                                if length == fsppg:
                                    one_second_list.append(dictionary)
                                elif length == fsppg * 60:
                                    one_minute_list.append(dictionary)
                    counter += 1
                # Calculating and printing the percentage of data saved from the corresponding record
                percentage_saved = percentage_saved / len(spo2_param) * 100
                print(str(percentage_saved) + "% of all segments of the record were saved.")
                if percentage_saved > 0:
                    patients_counter += 1

        # Message that indicates that the PPG signal information is missing for the current record
        if (len(ppg_header) == 0):
            print("The record lacks PPG signal information")
        # Message that indicates that the SpO2 information is missing for the current record
        if (len(spo2_header) == 0):
            print("The record lacks SpO2 information")

        print("////////////////////////////////////////////////////////")
    # Creating a Pandas dataframe out of the resultant list and printing the dataframe
    one_second_df = pd.DataFrame(one_second_list)
    one_minute_df = pd.DataFrame(one_minute_list)
    print(one_second_df)
    print(one_minute_df)
    print('ID:' + str(start_index) + ' - PATIENTS:' + str(patients_counter))

if __name__ == "__main__":
    main()