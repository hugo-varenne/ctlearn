import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import h5py
from h5py._hl.base import Empty

from ctapipe.core import run_tool
from ctapipe.tools.process import ProcessorTool
from ctapipe.io import EventSource
from ctapipe.calib import CameraCalibrator

from dl1_data_handler import image_mapper

from astropy.table import QTable
import astropy.io.misc.hdf5 as astropy_hdf5

# Enumeration containing the available image mappers values
DL1DH_IMAGE_MAPPERS = [
    "AxialMapper", "ShiftingMapper", "BilinearMapper", "BicubicMapper", "RebinMapper", "NearestNeighborMapper", "OversamplingMapper"
]

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

    def list_simtel_h5_files(self, simtel_dir):
        """
        List all Simtel files in the specified directory and returns a list of file paths.
        """
        simtel_files = [f for f in os.listdir(simtel_dir) if f.endswith('.simtel.h5')]
        return [os.path.join(simtel_dir, f) for f in simtel_files]

    def list_h5_files(self, simtel_dir):
        """
        List all H5 files in the specified directory and returns a list of file paths.
        """
        simtel_files = [f for f in os.listdir(simtel_dir) if f.endswith('.h5')]
        return [os.path.join(simtel_dir, f) for f in simtel_files]


    def extract_images(self, h5_file, mapper_class_name, extract_waveforms=False):
        """
        For each file, for each event, extract the DL1 image and transform it to a square image
        :param h5_file: list of (gammas | protons) files in H5 format
        :param mapper_class_name: name of the image mapper class ["AxialMapper", "ShiftingMapper", "BilinearMapper", "BicubicMapper", "RebinMapper", "NearestNeighborMapper", "OversamplingMapper"]
        :return: pandas dataframe
        """
        # Get mapper index from name
        mapper_index = DL1DH_IMAGE_MAPPERS.index(mapper_class_name)

        # For each gammas file, for each event, extract the DL1 image and transform it to a square image
        idx = 0
        table = None

        # Read all the events from the gamma file
        source = EventSource(h5_file, max_events=None)
        calib = CameraCalibrator(subarray=source.subarray)
        # Get camera geometry from first event
        sub = source.subarray
        geometry = sub.tel[1].camera.geometry
        # Create image mapper
        img_mapper = image_mapper.ImageMapper.from_name(mapper_class_name, geometry=geometry, subarray=sub)

        for event in source:
            # Perform basic calibration
            calib(event)

            for tel_id in sorted(event.r1.tel.keys()):
                # Get data to be stored in the table
                obs_id = source.obs_ids[0]
                event_id = event.index.event_id
                dl1_image = event.dl1.tel[tel_id].image
                if extract_waveforms:
                    r1_waveforms = event.r1.tel[tel_id].waveform
                else:
                    r1_waveforms = 0

                # Perform image mapping from hex to square
                dl1_image = np.expand_dims(dl1_image, axis=1)
                dl1dh_image = img_mapper.map_image(dl1_image)
                
                # Create a QTable with the extracted information
                if table == None:
                    table = QTable({
                    'obs_id': [obs_id], 
                    'event_id': [event_id],
                    'tel_id': [np.int8(tel_id)],
                    'dl1dh_image': [dl1dh_image[:, :, 0]],
                    'dl1dh_mapping': [np.int8(mapper_index)],
                    'r1_waveforms': [r1_waveforms[0, :, :] if extract_waveforms else 0]
                    })
                    table.meta['original_filename'] = os.path.basename(h5_file)
                else:
                    table.add_row([
                        [obs_id],
                        [event_id],
                        [np.int8(tel_id)],
                        [dl1dh_image[:, :, 0]],
                        [np.int8(mapper_index)],
                        [r1_waveforms]
                    ])

            idx += 1

        return table

    def convert_h5_to_dl1dh(self, h5_file, mapper_class_name, output_dir, extract_waveforms=False):
        """
        :brief Convert Simtel files to DL1DH files
        :param h5_files: list of (gammas | protons) files in H5 format
        :param mapper_class_name: name of the image mapper class ["AxialMapper", "ShiftingMapper", "BilinearMapper", "BicubicMapper", "RebinMapper", "NearestNeighborMapper", "OversamplingMapper"]
        :param output_dir: Output directory to store the DL1DH files
        """
        # Extract images
        data = self.extract_images(h5_file, mapper_class_name, extract_waveforms)
        # Write to HDF5 file
        dl1dh_filename = f"{Path(data.meta['original_filename']).stem}.dl1dh.h5"
        dl1dh_file_path = os.path.join(output_dir, dl1dh_filename)
        astropy_hdf5.write_table_hdf5(data, dl1dh_file_path, 'event/telescope/tel_001', overwrite=True)

        return dl1dh_file_path


    def extract_images_updated(self, 
        input_file,
        output_file
    ):
        """
        Create a reduced DL1 file keeping ctlearn & ctapipe compatibility
        """
        sections_to_keep = ['configuration', 'dl1', 'simulation']
        with h5py.File(input_file, "r") as fin, \
             h5py.File(output_file, "w") as fout:
    
            # Copy global attributes
            for key, value in fin.attrs.items():
                print(type(value))
                if isinstance(value, Empty):
                    value = ""
                elif isinstance(value, np.bytes_):
                    value = str(value, "utf-8")
                    print(value)
                
                fout.attrs[key] = value
    
            for path in sections_to_keep:
                if path in fin:
                    fin.copy(path, fout)
    
        return output_file


    def convert_h5_to_dl1dh_updated(self, h5_file, mapper_class_name, output_dir, extract_waveforms=False):
        """
        :brief Convert Simtel files to DL1DH files
        :param h5_files: list of (gammas | protons) files in H5 format
        :param mapper_class_name: name of the image mapper class ["AxialMapper", "ShiftingMapper", "BilinearMapper", "BicubicMapper", "RebinMapper", "NearestNeighborMapper", "OversamplingMapper"]
        :param output_dir: Output directory to store the DL1DH files
        """
        
        # Prepare Output
        dl1dh_filename = f"{Path(h5_file).name.split('.', 1)[0]}.h5"
        dl1dh_file_path = os.path.join(output_dir, dl1dh_filename)
        
        # Extract images
        self.extract_images_updated(h5_file, dl1dh_file_path)

        return dl1dh_file_path

    def read_h5_file(self, h5_file_path):
        """
        :brief Read an H5 file containing the DL1DH data
        :param h5_file_path: Path to the H5 file
        """
        rd_data = astropy_hdf5.read_table_hdf5(h5_file_path)
        return rd_data


def extract_images_of_files(input_dir, output_dir):
    # Get tools to convert simtel files to h5
    simtel_data_handler = SimtelDataHandler()
    files = simtel_data_handler.list_h5_files(input_dir)
    print(f"Found {len(files)} files")
    mapper_class_name = "BilinearMapper"
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.h5'):
            print(os.path.join(input_dir, filename))
            gammas_dl1dh_path = simtel_data_handler.convert_h5_to_dl1dh_updated(os.path.join(input_dir, filename), mapper_class_name, output_dir)
            print(f"Conversion complete for {gammas_dl1dh_path} file")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a range of files to another SSH server.")
    parser.add_argument("--input_dir", required=True, help="Input path containing H5 files with images to extract")
    parser.add_argument("--output_dir", required=True, help="Output path where to store H5 files with only the images")

    args = parser.parse_args()
    extract_images_of_files(args.input_dir, args.output_dir)