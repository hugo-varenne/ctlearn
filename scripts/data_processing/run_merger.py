
#!/usr/bin/env python

import argparse
import glob
import os
import re
import subprocess as sp
import numpy as np

def main():

    parser = argparse.ArgumentParser(
        description=("Script to run ctapipe-merge tool with DL1 hdf5 files"))
    parser.add_argument('--input_dir', '-i',
                        help='input directory',
                        default="./")
    parser.add_argument('--pattern', '-p',
                        help='pattern to mask unwanted files',
                        default="*.h5")
    parser.add_argument('--type',
                        help='string for the particle type',
                        default="gamma")
    parser.add_argument('--num_outputfiles', '-n',
                        help='number of output files',
                        default=10,
                        type=int)
    parser.add_argument('--output_dir', '-o',
                        help='output directory',
                        default="./")
    args = parser.parse_args()

    # Input handling
    abs_file_dir = os.path.abspath(args.input_dir)
    input = np.sort(glob.glob(os.path.join(abs_file_dir, args.pattern)))

    #half = len(input) // 2
    #input = input[half:]
    
    files_info = []
    run_numbers = []
    for file in input:
        number = [int(s) for s in re.findall(r'\d+', file.split("/")[-1])][-2]
        files_info.append((number, os.path.basename(file)))
            
    files_info.sort(key=lambda x: x[0])
    run_numbers = [r for (r, path) in files_info]
        
    print(len(run_numbers))

    print(run_numbers)
    n_runs = 0

    skip_broken_files = True
    for runs in np.array_split(run_numbers, args.num_outputfiles):
        output_file = f"{args.output_dir}/{args.type}_runs_{runs[0]}-{runs[-1]}.dl1.h5"
        cmd = [
             #"sbatch",
             #"-A",
             #"aswg",
             #"--output=/fefs/aswg/workspace/tjark.miener/allsky/slurm/slurm_%A_%a.out",
             #"--error=/fefs/aswg/workspace/tjark.miener/allsky/slurm/slurm_%A_%a.err",
             #"--partition=long",
             #"--partition=short",
             #"--mem-per-cpu=56G",
             #f"--job-name=testing_{str(analysis_id)}",
             "ctapipe-merge",
             "--overwrite",
             "-v",
             f"--MergeTool.skip_broken_files={skip_broken_files}",
             f"--output={output_file}",
             #"--provenance-log=/fefs/aswg/workspace/tjark.miener/allsky/slurm/provenance_log.log"
            ]

        for run in runs:
            filepath = next(path for (r, path) in files_info if r == run)
            cmd.append(f"{abs_file_dir}/{filepath}")
        print(f"Submitting merger tool to create: '{output_file}'")
        sp.run(cmd)
        print(cmd)

    return

if __name__ == "__main__":
    main()