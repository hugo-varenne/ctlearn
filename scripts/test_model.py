import sys
import os
import argparse
import importlib
import time
import json
import yaml
import shutil

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Input, Model

# CTLearn
import hdf5plugin, h5py
from tools.predict_model import MonoPredictCTLearnModel
from ctapipe.core import run_tool
import tools.predict_model
from traitlets.config import Config

# Others


class TestModel:

    def __init__(self, config_path):
        
        self.config = None   
        self.extract_config(config_path)
        self.model = os.path.join(self.config.training_model.TrainCTLearnModel.output_dir, "ctlearn_model.cpk")

    def extract_config(self, config_path):
        # Recursive function to extract nested information
        def recursive_config(d):
            # Convert nested dict in traitlets Config (adapted for CTLearn interaction)
            if isinstance(d, dict):
                cfg = Config()
                for k, v in d.items():
                    cfg[k] = recursive_config(v)
                return cfg
            return d
        # Get content of YAML file
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        # Apply recursive conversion to whole YAML content
        self.config = recursive_config(yaml_config)
        
    def test(self):
        # Verify if training and testing steps are defined
        if "training_model" in self.config and "testing_model" in self.config:
            config_training = self.config.training_model
            config_testing = self.config.testing_model
            # Prepare Prediction model
            # Predict on every test file
            self.testing_events = 0
            self.inference_time_global = 0
    
            # Define type of data to use for prediction
            if config_training.TrainCTLearnModel.reco_tasks == "type":
                particles = ["gammas_diffuse", "protons_diffuse"]
            else:
                particles = ["gammas_point"]
            # Create / Remove prediction folder
            shutil.rmtree(os.path.join(config_training.TrainCTLearnModel.output_dir, "predict"), ignore_errors=True)
            os.makedirs(os.path.join(config_training.TrainCTLearnModel.output_dir, "predict"), exist_ok=True)
            for particle in particles:
                directory = os.path.join(config_testing.data_path, particle, "test")
                for filename in os.listdir(directory):
                    if filename.endswith(".h5"):
                        # Prepare new filename as output
                        predict_file = os.path.basename(filename).split(".", 1)[0]
                        input_url = os.path.join(directory, filename)
                        # Create results file
                        os.makedirs(os.path.join(config_training.TrainCTLearnModel.output_dir, "predict"), exist_ok=True)
                        output_url = os.path.join(config_training.TrainCTLearnModel.output_dir, "predict", f"{predict_file}.h5")
                        # Initialize the prediction by choosing correct prediction format
                        match config_training.TrainCTLearnModel.reco_tasks:
                            case "type":
                                if config_testing.PredictCTLearnModel.DLImageReader.mode == "stereo":
                                    model_predict = StereoPredictCTLearnModel(input_url=input_url, load_type_model_from=self.model, output_path=output_url, config=config_testing)   
                                else:
                                    model_predict = MonoPredictCTLearnModel(input_url=input_url, load_type_model_from=self.model, output_path=output_url)
                            case "energy":
                                if config_testing.PredictCTLearnModel.DLImageReader.mode == "stereo":
                                    model_predict = StereoPredictCTLearnModel(input_url=input_url, load_energy_model_from=self.model, output_path=output_url, config=config_testing)   
                                else:
                                    model_predict = MonoPredictCTLearnModel(input_url=input_url, load_energy_model_from=self.model, output_path=output_url)
                            case "skydirection":
                                model_predict = StereoPredictCTLearnModel(input_url=input_url, load_skydirection_model_from=self.model, output_path=output_url, config=config_testing)
                            case "cameradirection":
                                model_predict = MonoPredictCTLearnModel(input_url=input_url, load_cameradirection_model_from=self.model, output_path=output_url)
                            case _:
                                print("ERROR")
                        # Run the prediction
                        start = time.time()
                        try:
                            model_predict.run()
                        except SystemExit as e:
                            print(f"Caught SystemExit ({e.code}, continuing...)")
                        stop = time.time()
    
                        # Add value for each file to metrics
                        self.testing_events += model_predict.dl1dh_reader._get_n_events()
                        self.inference_time_global += (stop - start) * 1000 # ms
        else:
            print("No training and testing steps planned for testing the model")

    def save_metrics(self):
        # Verify if training step is defined
        if "training_model" in self.config:
            config_training = self.config.training_model
            # Load JSON file without erasing content (file shared for training and testing)
            json_path = os.path.join(config_training.TrainCTLearnModel.output_dir, "metrics.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}
                
            # Add metrics to JSON
            data["testing_events"] = self.testing_events
            data["testing_ms"] = self.inference_time_global
    
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            print("No training step planned for saving metrics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a custom model with provided parameters")
    parser.add_argument("--yaml_file", required=True, help="Yaml file for testing information")

    args = parser.parse_args()
    
    testing = TestModel(args.yaml_file)
    testing.test()
    testing.save_metrics()
