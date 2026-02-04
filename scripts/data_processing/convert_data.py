import os
import argparse
from pathlib import Path

from ctapipe.core import run_tool
from ctapipe.tools.process import ProcessorTool

class SimtelDataHandler:
    """
    A class to handle Simtel files
    """
    def __init__(self):
        """
        Initializes the SimtelDataHandler instance.
        """
        pass

    def convert_simtel_to_h5(self, simtel_dir, simtel_filename):
        """
        Converts a Simtel file to an H5 file using the ctapipe ProcessorTool.
        """
        # Construct full paths for the input Simtel file and output H5 file
        simtel_file = os.path.join(simtel_dir, simtel_filename)
        h5_file = f"{Path(simtel_file).stem}.h5"
    
        # Prepare command-line arguments for the ProcessorTool
        argv = [
            f"--input={simtel_file}",
            f"--output={h5_file}",
            f"--progress",
            f"--DataWriter.write_r1_waveforms=True",
            f"--DataWriter.write_dl1_images=True",
            f"--DataWriter.write_dl1_parameters=True",
            f"--DataWriter.Contact.name=LG"
        ]
    
        # Run the ProcessorTool with the specified arguments
        try:
            run_tool(ProcessorTool(), argv=argv, cwd=simtel_dir)
            print(f"Successfully converted {simtel_file} to {h5_file}")
        except Exception as e:
            print(f"Error converting {simtel_file} to H5: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a range of files to another SSH server.")
    parser.add_argument("--folder", required=True, help="Folder path to convert in H5 files")

    args = parser.parse_args()
    
    data_handler = SimtelDataHandler()
    for root, dirs, files in os.walk(args.folder):
        for filename in files:
            if filename.endswith('.simtel.gz'):
                data_handler.convert_simtel_to_h5(root, filename)

    