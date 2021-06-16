import numpy as np
import time
import sys, os
import shutil
import traceback
import json_processor
from multiprocessing.pool import Pool

PROCESSES = 12
TESTING = False
FOLDER = ''

def get_json_filenames():
    filename_list = []
    for entry in os.scandir(f'/mnt/d/rl_ai/data/{FOLDER}/next_json'):
        if entry.path.endswith(".json") and entry.is_file():
            filename_list.append(entry.path)
    return filename_list

def add_json_to_pool(filename):
    processing_list.append(pool.apply_async(json_processor.process_replay, (filename,)))
    associated_filenames.append(filename)
    filename_list.remove(filename)

def fetch_result_from_pool():
    try:
        process = processing_list[0]
        output = process.get(timeout=40)
        output_arrays.append(output)
        processing_list.remove(process)
        old_path = associated_filenames[0]
        new_path = f'/mnt/d/rl_ai/data/{FOLDER}/processed_json/' + \
            old_path.split('/')[7]
        if not TESTING: shutil.move(old_path, new_path)
        associated_filenames.remove(associated_filenames[0])
    except:
        traceback.print_exc()
        print('THE FILE {} ENCOUNTERED AN EXCEPTION [OR TOOK TOO LONG] AND WILL NOT BE ADDED TO THE DATASET'.format(
            associated_filenames[0].split('/')[1]))
        if process in processing_list:
            processing_list.remove(process)
        associated_filenames.remove(associated_filenames[0])

if __name__ == "__main__":

    pool = Pool(processes=PROCESSES)
    processing_list = []
    associated_filenames = []
    output_arrays = []

    FOLDER = sys.argv[1]
    path = f'/mnt/d/rl_ai/data/{FOLDER}/'
    if not os.path.isdir(path):
        sys.exit(f'THE FOLDER \'{FOLDER}\' COULD NOT BE FOUND')

    with pool:

        filename_list = get_json_filenames()
        print(len(filename_list))

        while len(filename_list) != 0:
            for filename in filename_list:
                if len(processing_list) < PROCESSES:
                    add_json_to_pool(filename)
                    print(len(filename_list), len(processing_list))
                else:
                    fetch_result_from_pool()

        while len(processing_list) != 0:
            for process in processing_list:
                fetch_result_from_pool()
                print(len(filename_list), len(processing_list))

    print('merging output arrays...')
    all_output = np.concatenate(output_arrays, axis=0)
    # all_output = np.unique(all_output,axis=0) # remove duplicate rows

    print('writing dataset to disk...')
    if TESTING: np.savetxt('/mnt/d/rl_ai/data/{FOLDER}/npy/test.csv', all_output, delimiter=',')
    else: np.save(f'/mnt/d/rl_ai/data/{FOLDER}/npy/{FOLDER}_' + str(int(time.time()) % 1000000), all_output)

    print('finished.\nrows in dataset partition: {}'.format(all_output.shape[0]))