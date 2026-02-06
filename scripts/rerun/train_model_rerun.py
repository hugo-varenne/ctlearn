# System
import sys
import os
import argparse
import time
import json
import yaml
import importlib
import shutil

# Tensorflow
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.keras import Input, Model
import tensorflow_model_optimization as tfmot

# CTLearn
import hdf5plugin, h5py
from ctlearn.core.model import CTLearnModel
from tools.train_model_rerun import TrainCTLearnModel
from traitlets.config import Config

# Logs
import warnings
from astropy.utils.metadata.exceptions import MergeConflictWarning
# Prevent type of logs
warnings.filterwarnings("ignore", category=MergeConflictWarning)

# Others (rerun)
import signal
def handle_shutdown(signum, frame):
    print(f"Received signal {signum}, saving checkpoint...", flush=True)
    sys.exit(0)


signal.signal(signal.SIGUSR1, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

class TrainModel:

    def __init__(self, config_path, custom_path=None):
        
        self.config = None
        self.config_path = config_path
        self.extract_config(config_path)
        if not os.path.exists(os.path.join(self.config.training_model.TrainCTLearnModel.output_dir, "interrupt")):
            self.prepare_model(custom_path)

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
        
    def prepare_model(self, custom_path):
        # Verify if prepare step is defined
        if "prepare_model" in self.config:
            prepare_config = self.config.prepare_model
            if custom_path and "CustomModel" in prepare_config:
                module = importlib.import_module(f"templates.{prepare_config.CustomModel.model_filename}")
                CustomModel = getattr(module, prepare_config.CustomModel.model_name)
                model = CustomModel(input_shape=prepare_config.image_shape, reco_task=prepare_config.tasks[0], num_classes=prepare_config.num_classes, name="custom_model")
                if "type" in prepare_config.tasks:
                    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                else:
                    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
                model.save(prepare_config.temp_dir)
            else:   
                self.temp_model = CTLearnModel.from_name(prepare_config.model_type, 
                                                         input_shape=tuple(prepare_config.input_shape), 
                                                         tasks=prepare_config.tasks, 
                                                         config=prepare_config).model
                self.temp_model.save(prepare_config.temp_dir)
        else:
            print("No preparation step planned for this model")

    def train(self):
        # Verify if training step is defined
        if "training_model" in self.config:
            config_training = self.config.training_model
            # Remove existing model at location if necessary (check if currently runned)
            interupt = os.path.join(config_training.TrainCTLearnModel.output_dir, "interrupt")
            if os.path.exists(config_training.TrainCTLearnModel.output_dir) and not os.path.exists(interupt):
                shutil.rmtree(config_training.TrainCTLearnModel.output_dir)
            # Create CTLearnModel instance + Setup
            model = TrainCTLearnModel(config=config_training)
            # Start timer for time of training
            start = time.time()
            try:
                # Run training
                model.run()
            except SystemExit as e:
                print(f"Caught SystemExit ({e.code}, continuing...)")
            end = time.time()
            # Define metrics
            self.training_ms = (end - start) * 1000 # ms
            self.training_events = model.dl1dh_reader._get_n_events()
    
            # Restore model state if pruning
            if "pruning_model" in config_training.TrainCTLearnModel:
                pruned_model_path = os.path.join(config_training.TrainCTLearnModel.output_dir, "ctlearn_model.cpk")
                # Load pruned model
                with tfmot.sparsity.keras.prune_scope():
                    pruned_model = tf.keras.models.load_model(pruned_model_path)
                # Strip model of pruning weights
                strip_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
                # Save stripped model
                strip_model.save(pruned_model_path)              
        else:
            print("No training step planned for this model")
            
    def save_config(self):
        # Verify if training step is defined
        if "training_model" in self.config:
            # Create output folder + copy config
            if os.path.exists(self.config.training_model.TrainCTLearnModel.output_dir):
                shutil.copy(self.config_path, os.path.join(self.config.training_model.TrainCTLearnModel.output_dir, "config.yaml"))
        else:
            print("No training step planned for this model")

    def save_metrics(self):
        # Verify if training step is defined
        if "training_model" in self.config:
            config_training = self.config.training_model
            # Load model in keras format
            keras_model = tf.keras.models.load_model(os.path.join(config_training.TrainCTLearnModel.output_dir, "ctlearn_model.cpk"))
    
            # Calculate some metrics based on the model
            self.num_params = keras_model.count_params() 
            self.num_layers = len(self.get_atomic_layers(keras_model))
            self.estimated_flops = self.get_flops(keras_model)
            
            # Load JSON file without erasing content (file shared for training and testing)        
            json_path = os.path.join(config_training.TrainCTLearnModel.output_dir, "metrics.json")
            # Verify if the JSON file for metrics exists
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}
                
            # Add metrics to JSON
            data["num_params"] = self.num_params
            data["training_ms"] = self.training_ms
            data["training_events"] = self.training_events
            data["layers"] = self.num_layers
            data["estimated_flops"] = self.estimated_flops
            # Save metrics in the JSON file for metrics
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            print("No training step planned for saving metrics")

    def get_flops(self, model):
        input_data = tf.random.normal([1] + self.config.prepare_model.input_shape)
        # Convert model to graph function
        concrete_func = tf.function(model).get_concrete_function(input_data)
        
        # Get graph definition
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()
        
        # Use profiler to calculate FLOPs (and not verbose)
        run_meta = tf.compat.v1.RunMetadata()
        builder = tf.compat.v1.profiler.ProfileOptionBuilder
        opts = builder(builder.float_operation()).build()
        opts['output'] = 'none'

        # Extract flops
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )
        
        # Check if it works (if not value of total flops is 0
        if flops is not None:
            return flops.total_float_ops / 1e9
        else:
            return 0

    def get_atomic_layers(self, model):
        visited = set()
        layer_list = []
        
        def _collect(layer):
            if id(layer) in visited:
                return
            visited.add(id(layer))
        
            # Skip InputLayer
            if isinstance(layer, tf.keras.layers.InputLayer):
                return
        
            # Recursively explore submodels without counting the model itself
            if isinstance(layer, tf.keras.Model):
                for sublayer in layer.layers:
                    _collect(sublayer)
                return
        
            # Explore attributes for inner layers (custom blocks)
            for attr_name in dir(layer):
                try:
                    attr = getattr(layer, attr_name)
                except Exception:
                    continue
                if isinstance(attr, tf.keras.layers.Layer):
                    _collect(attr)
                elif isinstance(attr, (list, tuple)):
                    for obj in attr:
                        if isinstance(obj, tf.keras.layers.Layer):
                            _collect(obj)
        
            # Count layer if it has no sublayers (atomic)
            if not hasattr(layer, "layers") or len(getattr(layer, "layers", [])) == 0:
                layer_list.append((layer.name, layer.__class__.__name__))
        
        # Start from top-level layers
        for l in model.layers:
            _collect(l)
        
        return layer_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cd")
    parser.add_argument("--yaml_file", required=True, help="Yaml file for training information")
    parser.add_argument("--custom", required=False, help="Class file containing model class.")

    args = parser.parse_args()
    
    training = TrainModel(args.yaml_file, args.custom)
    training.train()
    training.save_config()
    training.save_metrics()