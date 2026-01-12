# System
import sys
import os
import argparse
import base64
from io import BytesIO, StringIO
import importlib
import time
import json
from datetime import datetime
import yaml

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Input, Model

# Graphs and metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, recall_score, f1_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.ticker as ticker
from sklearn.calibration import calibration_curve

# CTLearn
import hdf5plugin, h5py
import ctaplot
import astropy.units as u
from traitlets.config import Config


class Report:

    def __init__(self, model_path, config_path, output_path, task="type", name="Report"):
        self.config = None   
        self.extract_config(config_path)
        
        self.model_path = model_path
        self.output_path = output_path
        self.task = task
        self.report_name = name
        self.metrics = []
        
        # Define the type of model
        match task:
            case "type":
                self.threshold = 0.5
                self.model_type = "Particle classification"
            case "energy":
                self.model_type = "Energy regression"
            case "cameradirection":
                self.model_type = "Camera Direction regression"
            case "skydirection":
                self.model_type = "Sky Direction regression"
            case _:
                self.model_type = "Unknown task"

        # Procedure to execute to build the report
        self.create_experiment_folder()
        self.load_model(self.model_path)
        self.training_plot()
        self.get_model_summary()
        self.get_prediction_data(task)
        
        # Depending on task, prepare different graphs
        if task == "type":
            self.get_type_graphs()
        elif task == "energy":
            self.get_energy_graphs()
        elif task == "cameradirection" or task == "skydirection":
            self.get_direction_graphs(task)
        # Extract metrics
        self.extract_performance_metrics()
        self.extract_evaluation_metrics()
        self.build_report()

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
        
    def load_model(self, model_path):
        # Load the model in the class for future usage
        model_file = os.path.join(model_path, "ctlearn_model.cpk")
        self.model = tf.keras.models.load_model(model_file)

    def build_report(self):
        # Verify if training step is defined in the configuration
        if "training_model" in self.config:
            config_training = self.config.training_model
        else:
            print("No training configuration provided for this model")
            
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <title>Model Report</title>
        <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f8f9fa;
            color: #333;
            margin: 2rem auto;
            max-width: 900px;
            line-height: 1.6;
        }}
        h1 {{
            text-align: center;
            color: #0078d7;
        }}
        h2 {{
            text-align: center;
            font-size: 3.0rem;
        }}
        section {{
            background: #fff;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem 2rem;
            justify-content: center;  
            max-width: 80%;           
            margin: 0 auto;
        }}
        .col p {{
            margin: 0.4rem 0;
            text-align: center;
        }}
        pre {{
            background: #f0f0f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }}
        .metric {{
            font-size: 1.1rem;
            margin: 0.5rem 0;
        }}
        img {{
            display: block;
            margin: 1rem auto;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}
        footer {{
            text-align: center;
            color: #666;
            font-size: 0.9rem;
            margin-top: 2rem;
        }}
        </style>
        </head>
        <body>
        <h1>TensorFlow Model Report</h1>
        
        <section>
        <h2>General information</h2>
        <div class="grid">
        <div class="col">
        <p><strong>Generated on :</strong> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}</p>
        <p><strong>Number of Epochs :</strong> {self.effective_epochs} (Initially {config_training.TrainCTLearnModel.n_epochs} epochs)</p>
        {f'<p class="metric"><strong>Number of Training Events :</strong> {self.metrics["training_events"]}</p>' if "training_events" in self.metrics else ""}
        </div>
        <div class="col">
        <p><strong>Task :</strong> {self.model_type}</p>
        <p><strong>Batch Size :</strong> {config_training.TrainCTLearnModel.batch_size}</p>
        {f'<p class="metric"><strong>Number of Testing Events :</strong> {self.metrics["testing_events"]}</p>' if "testing_events" in self.metrics else ""}
        </div>
        </div>
        </section>
        
        <section>
        <h2>Performance Metrics</h2>
        <div class="grid">
        <div class="col">
        {f'<p class="metric"><strong>Number of Parameters :</strong> {self.metrics["num_params"]}</p>' if "num_params" in self.metrics else ""}
        {f'<p class="metric"><strong>Number of Layers :</strong> {self.metrics["layers"]}</p>' if "layers" in self.metrics else ""}
        {f'<p class="metric"><strong>Training Time (s) :</strong> {(self.metrics["training_ms"]/1000):.3f}</p>' if "training_ms" in self.metrics else ""}
                </div>
        <div class="col">
        {f'<p class="metric"><strong>Total Inference Time (s) :</strong> {(self.metrics["testing_ms"]/1000):.3f}</p>' if "testing_ms" in self.metrics else ""}
        {f'<p class="metric"><strong>Inference Time per Events (ms) :</strong> {(self.metrics["testing_ms"]/self.metrics["testing_events"]):.3f}</p>' if "testing_ms" in self.metrics and "testing_events" in self.metrics else ""}
        {f'<p class="metric"><strong>Estimated Flops :</strong> {self.metrics["estimated_flops"]:.3f} (GFLOPs)</p>' if "estimated_flops" in self.metrics else ""}
         </div>
        </div>
        <h2>Evaluation Metrics</h2>
        {self.evaluation_metrics}
        </section>
        
        <section>
        <h2>Model Summary</h2>
        <pre>{self.model_summary}</pre>
        </section>
        
        <section>
        <h2>Graphics</h2>
        <img src="data:image/png;base64,{self.training_plot}" alt="Prediction Distribution Plot">
        {self.additional_graphs}
        </section>
        
        <footer>
        Model Generated using CTLearn library.
        </footer>
        
        </body>
        </html>
        """
        # Write the html file in the dedicated space
        output_file = os.path.join(self.output_path, f"{self.report_name}.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        
        print(f"✅ HTML report generated: {output_file}")


    def create_experiment_folder(self):
        os.makedirs(self.output_path, exist_ok=True)
        
    #########################################################################    
    ######################### DEFAULT GRAPHICS CASE #########################
    ######################################################################### 
    
    def training_plot(self):
        # Extract training information from file
        training_results = os.path.join(self.model_path, "training_log.csv")
        if os.path.exists(training_results):
            loss = pd.read_csv(training_results)
            # Get actual number of epochs trained
            self.effective_epochs = len(loss)
            plt.figure(figsize=(8,5))
            
            # Plot losses
            plt.plot(loss["epoch"], loss["loss"], label="Training Loss")
            plt.plot(loss["epoch"], loss["val_loss"], label="Validation Loss")

            # Prepare graphic visualization
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss", fontsize=20, fontweight=500)
            plt.legend()
            plt.grid(True)
            
            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            self.training_plot = base64.b64encode(buf.read()).decode("utf-8")           
        else: 
            print("No training performed")

    def get_model_summary(self):
        # Extract in string format the model summary
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        self.model_summary = stream.getvalue()

    ###################################################################    
    ######################### METRICS DISPLAY #########################
    ###################################################################
    
    def extract_performance_metrics(self):
        # Load file containing metrics and save it
        metrics_path = os.path.join(self.model_path, "metrics.json")
        if os.path.exists(metrics_path):
                with open(metrics_path, "r") as file:
                    self.metrics = json.load(file)
    
    def extract_evaluation_metrics(self):
        if not self.predictions is None :
            if self.task == "energy":
                # Extract necessary information to calculate metrics
                predicted_values = self.predictions["CTLearn_energy"]
                ground_truth = self.predictions["true_energy"]
                self.evaluation_metrics = f"""
                <div class="grid">
                    <div class="col">
                        <p class="metric"><strong>MSE :</strong> {mean_squared_error(predicted_values, ground_truth):.4f}</p>
                        <p class="metric"><strong>Coefficient of Determination (R²) :</strong> {r2_score(predicted_values, ground_truth):.4f}</p>
                    </div>
                    <div class="col">
                        <p class="metric"><strong>RMSLE :</strong> {np.sqrt(tf.reduce_mean(np.square(ground_truth - predicted_values))):.4f}</p>
                        <p class="metric"><strong>MAE :</strong> {mean_absolute_error(predicted_values, ground_truth):.4f}</p>
                    </div>
                </div>
                """            
            elif self.task == "cameradirection" or self.task == "skydirection":
                # Extract necessary information to calculate metrics
                alt_pred = self.predictions["CTLearn_alt"]
                az_pred  = self.predictions["CTLearn_az"]
                alt_true = self.predictions["true_alt"]
                az_true  = self.predictions["true_az"]
                # Calculate absolute difference
                diff = np.abs(az_true - az_pred)
                # Get angular error in degree
                directional_errors = self.directional_error_deg(az_true, alt_true,
                                               az_pred, alt_pred)
                self.evaluation_metrics = f"""
                <div class="grid">
                    <div class="col">
                        <p class="metric"><strong>MDAE :</strong> {np.mean(directional_errors):.4f}</p>
                        <p class="metric"><strong>MAAE Circular :</strong> {np.mean(np.minimum(diff, 360-diff)):.4f}</p>
                    </div>
                    <div class="col">
                        <p class="metric"><strong>RMSDE :</strong> {np.sqrt(np.mean(directional_errors**2)):.4f}</p>
                        <p class="metric"><strong>MAAE Linear :</strong> {np.mean(diff):.4f}</p>
                    </div>
                </div>
                """            
            elif self.task == "type":

                # Extract necessary information to calculate metrics
                predicted_values = self.predictions.apply(self.apply_corrected_predictions, axis=1)
                ground_truth = self.predictions["particle_id"]
                
                self.evaluation_metrics = f"""
                <div class="grid">
                    <div class="col">
                        <p class="metric"><strong>Accuracy :</strong> {accuracy_score(ground_truth, predicted_values):.4f}</p>
                        <p class="metric"><strong>Recall :</strong> {recall_score(ground_truth, predicted_values):.4f}</p>
                    </div>
                    <div class="col">
                        <p class="metric"><strong>F1 Score :</strong> {f1_score(ground_truth, predicted_values):.4f}</p>
                        <p class="metric"><strong>Brier Score :</strong> {brier_score_loss(ground_truth, self.predictions["CTLearn_prediction"]):.4f}</p>
                    </div>
                </div>
                """     
        else:
            print("No prediction Data found")
            
    ######################################################################        
    ######################### GET PREDICTED DATA #########################
    ###################################################################### 
    
    def get_prediction_data(self, task):
        prediction_directory = os.path.join(self.model_path, "predict")
    
        # Prepare DataFrame and mapping for classes
        self.predictions = pd.DataFrame()
        mapping = {0: 1, 101: 0}
        mapping_name = {0: "Gamma", 101: "Proton"}
        
        # Process prediction elements
        for file in os.listdir(prediction_directory):
                filename = os.path.join(prediction_directory, file)
                if filename.endswith(".h5"):
                    with h5py.File(filename, 'r') as data:
                        # Get DL2 predictions
                        if task == "type" and 'dl2/event/subarray/classification/CTLearn' in data:
                            # Predictions
                            pred_ds = data['dl2/event/subarray/classification/CTLearn']['event_id', 'CTLearn_prediction', 'CTLearn_is_valid']

                        elif task == "energy" and 'dl2/event/subarray/energy/CTLearn' in data:
                            # Predictions
                            pred_ds = data['dl2/event/subarray/energy/CTLearn']['event_id', 'CTLearn_energy', 'CTLearn_is_valid', 'CTLearn_energy_uncert']

                        elif task == "cameradirection" and 'dl2/event/subarray/geometry/CTLearn' in data:
                            # Predictions
                            pred_ds = data['dl2/event/subarray/geometry/CTLearn']['event_id', 'CTLearn_alt', 'CTLearn_az', 'CTLearn_is_valid']
                           
                        # Take only valid events
                        pred_ds = pd.DataFrame(pred_ds)
                        pred_ds = pred_ds[pred_ds["CTLearn_is_valid"] == 1]
                        
                        # Ground Truth
                        ground_truth = data["simulation/event/subarray/shower"][:]
                        ground_truth_df = pd.DataFrame(ground_truth, columns=["event_id", "true_az", "true_alt", "true_energy", "true_shower_primary_id"])
                        # Combine ground truth with prediction
                        combined_df = pd.merge(pred_ds, ground_truth_df, on="event_id", how="inner")
                        combined_df["particle_id"] = combined_df["true_shower_primary_id"].map(mapping)
                        combined_df["particle_name"] = combined_df["true_shower_primary_id"].map(mapping_name)
                        
                        # Get telescope information(camera coordinates)
                        pointing_coordinates = data["dl1/monitoring/telescope/pointing/tel_001"]["altitude", "azimuth"]
                        combined_df["altitude"] = np.unique(pointing_coordinates["altitude"])[0]
                        combined_df["azimuth"] = np.unique(pointing_coordinates["azimuth"])[0]
                        
                        # Merge with others result files
                        self.predictions = pd.concat([self.predictions, combined_df], ignore_index=True)
                        
    ##########################################################################
    ######################### SPECIFIC GRAPHICS CASE #########################
    ##########################################################################
    
    def get_type_graphs(self):
        html_section = """"""
        confusion_matrix = self.confusion_matrix()
        html_section += f"""        
        <img src="data:image/png;base64,{confusion_matrix}" alt="Confusion Matrix">
        """
        roc_curve = self.roc_curve()
        html_section += f"""        
        <img src="data:image/png;base64,{roc_curve}" alt="ROC Curve">
        """
        calibration_curve = self.calibration_curve()
        html_section += f"""        
        <img src="data:image/png;base64,{calibration_curve}" alt="Calibration Curve">
        """
        distribution = self.gammaness_distribution()
        html_section += f"""
        <img src="data:image/png;base64,{distribution}" alt="Gammaness Distribution graphic">
        """
        
        self.additional_graphs = html_section
    
    def get_energy_graphs(self):
        html_section = """"""
        distribution = self.energy_distribution()
        html_section += f"""
        <img src="data:image/png;base64,{distribution}" alt="Energy Distribution graphic">
        """
        migration_matrix = self.migration_matrix()
        html_section += f"""
        <img src="data:image/png;base64,{migration_matrix}" alt="Migration Matrix graphic">
        """
        energy_resolution = self.energy_resolution()
        html_section += f"""
        <img src="data:image/png;base64,{energy_resolution}" alt="Energy Resolution graphic">
        """
        energy_bias = self.energy_bias()
        html_section += f"""
        <img src="data:image/png;base64,{energy_bias}" alt="Energy Bias and Standard deviation graphic">
        """
               
        self.additional_graphs = html_section

    def get_direction_graphs(self, task):
        html_section = """"""
        precision_in_degrees = self.precision_in_degrees()
        html_section += f"""
        <img src="data:image/png;base64,{precision_in_degrees}" alt="Precision within X degrees graphic">
        """        
        distribution = self.angular_distribution()
        html_section += f"""
        <img src="data:image/png;base64,{distribution}" alt="Altitude-Azimuth Distribution graphic">
        """
        angular_resolution = self.angular_resolution()
        html_section += f"""
        <img src="data:image/png;base64,{angular_resolution}" alt="Angular Resolution graphic">
        """
        angular_bias = self.angular_bias()
        html_section += f"""
        <img src="data:image/png;base64,{angular_bias}" alt="Angular Bias and Standard deviation graphic">
        """
        
        self.additional_graphs = html_section
        
    ##################################################################################
    ######################### PARTICLE CLASSIFICATION GRAPHS #########################
    ##################################################################################

    def confusion_matrix(self):
        if not self.predictions is None :   
            # Extract necessary information
            predicted_values = self.predictions.apply(self.apply_corrected_predictions, axis=1)
            ground_truth = self.predictions["particle_id"]
            labels = self.predictions["particle_name"].unique()

            # Set figure dimensions
            plt.figure(figsize=(7,7))
            
            # Generate Confusion matrix
            cm = confusion_matrix(ground_truth, predicted_values)
            
            # Prepare graphic
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.ylabel("Ground truth")
            plt.xlabel("Predictions")
            plt.xticks(np.arange(-0.5, cm.shape[1], 0.5), minor=True)
            plt.yticks(np.arange(-0.5, cm.shape[0], 0.5), minor=True)
            plt.grid(which="minor", color="black", linewidth=0.5)

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        else:
            print("No prediction Data found")
    
    def roc_curve(self):
        if not self.predictions is None :
            # Calculate ROC and AUC
            fpr_gamma, tpr_gamma, thresholds_gamma = metrics.roc_curve(self.predictions["particle_id"], self.predictions['CTLearn_prediction'], pos_label=1)
            auc_gamma = metrics.auc(fpr_gamma, tpr_gamma)

            # Set figure dimensions
            plt.figure(figsize=(7,7))
            
            # Plot ROC
            plt.plot(fpr_gamma, tpr_gamma, label=f"Gamma vs Proton (AUC={auc_gamma:.3f})")

            # Prepare graphic visualization
            plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve: Gamma vs Proton", fontsize=20, fontweight=500)
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.6)

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        else:
            print("No prediction Data found")

    def calibration_curve(self):
        if not self.predictions is None :
            # Extract necessary information
            predicted_values = self.predictions["CTLearn_prediction"]
            ground_truth = self.predictions["particle_id"]

            # Set figure dimensions
            plt.figure(figsize=(7,7))
            
            # Get fractions of positives
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ground_truth, predicted_values, n_bins=100, strategy="uniform"
            )

            # Prepare graphic
            plt.plot(mean_predicted_value, fraction_of_positives, label=f"{self.model_type}")
            plt.plot([0, 1], [0, 1], "--", label="Best Calibration Curve", color="black")       
            plt.title("Probability Calibration Curve")
            plt.xlabel("Probability [%]")
            plt.ylabel("Fraction of positives [%]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        else:
            print("No prediction Data found")

    def gammaness_distribution(self):
        if not self.predictions is None :
            # Get uniques particles
            particles_type = self.predictions["particle_name"].unique()

            # Set figure dimensions
            plt.figure(figsize=(7,7))
            
            # Display points for each particle type
            for particle in particles_type:
                particle_data = self.predictions[self.predictions["particle_name"] == particle]
                plt.hist(
                    particle_data["CTLearn_prediction"],
                    bins=100,
                    range=(0, 1),
                    histtype="step",
                    density=True,
                    label=particle,
                )
            # Prepare graphic
            plt.title("Gammaness Distribution", fontsize=20, fontweight=500)
            plt.xlabel("Gammaness")
            plt.ylabel("Density")
            plt.legend()
            
            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        else:
            print("No prediction Data found")
            
    ############################################################################        
    ######################### ENERGY REGRESSION GRAPHS #########################
    ############################################################################ 
    
    def energy_distribution(self):
        if not self.predictions is None :
            # Get uniques particles
            particles_type = self.predictions["particle_name"].unique()

            # Set figure dimensions
            plt.figure(figsize=(7,7))
        
            # Prepare bins for graph
            log_bins = np.logspace(
                np.log10(self.predictions["CTLearn_energy"].min()),
                np.log10(self.predictions["CTLearn_energy"].max()),
                100,
            )
            
            # Display points for each particle type
            for particle in particles_type:
                particle_data = self.predictions[self.predictions["particle_name"] == particle]
        
                # Build histogram
                plt.hist(
                    particle_data["CTLearn_energy"],
                    bins=log_bins,
                    range=(0, 1),
                    histtype="step",
                    density=True,
                    label=particle,
                )
                
            # Show graphic
            plt.xlabel("Energy [TeV]")
            plt.ylabel("Density")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(title="Particle Types")
            plt.title("Energy Distribution", fontsize=20, fontweight=500)
            plt.grid(alpha=0.3)

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8") 
        else:
            print("No prediction Data found")

    def migration_matrix(self):
        if not self.predictions is None :
            particles_type = self.predictions["particle_name"].unique()
            
            # Set figure dimensions
            fig, axes = plt.subplots(1, len(particles_type), figsize=(4 * len(particles_type) , 3.5))
            
            for i, particle in enumerate(particles_type):
                # Set the graph differently based on number of particles
                if len(particles_type) > 1:
                    ax = axes[i]
                else:
                    ax = axes
                    
                # Extract only wanted energies
                particle_df = self.predictions[self.predictions["particle_name"] == particle]
                
                # Extract only wanted energies
                energy_pred = particle_df["CTLearn_energy"]
                energy_true = particle_df["true_energy"]
                
                # Extract bins using a logarithm space
                log_bins = np.logspace(
                    np.log10(
                        min(
                            (
                                min(energy_true),
                                min(energy_pred),
                            )
                        )
                    ),
                    np.log10(
                        max(
                            max(energy_true),
                            max(energy_pred),
                        )
                    ),
                    # Number of bins
                    100,
                )
                
                # Prepare perfect prediction line
                ax.plot(
                    [log_bins[0], log_bins[-1]],
                    [log_bins[0], log_bins[-1]],
                    color="#cf004b",
                    ls="--",
                )
                
                # Plot the data according to bins
                ax.hist2d(
                    energy_true,
                    energy_pred,
                    bins=log_bins,
                    cmap="viridis",
                    norm=plt.cm.colors.LogNorm(),
                )

                # Prepare the graph styling
                ax.set_xlim(log_bins[0], log_bins[-1])
                ax.set_ylim(log_bins[0], log_bins[-1])
                ax.set_xlabel("True energy [TeV]")
                ax.set_ylabel("Predicted energy [TeV]")
                ax.set_title(f"{particle}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                # Add a color bar to show repartition of values
                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label("Number of events")

            # Prepare Global graph
            fig.suptitle(f"Migration Matrix", fontsize=20, fontweight=500)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")  
        else:
            print("No prediction Data found")

    def energy_resolution(self):
        if not self.predictions is None :
            # Extract only wanted energies
            energy_pred = self.predictions["CTLearn_energy"]
            energy_true = self.predictions["true_energy"]

            # Build bins (0.1–100 TeV)
            log_bins = (
                np.logspace(
                    np.log10(min(energy_true)),
                    np.log10(max(energy_true)),
                    num=int(
                        np.log10(max(energy_true) / min(energy_true))
                        * 5
                    )
                    + 1,
                )
                * u.TeV
            )

            # Get bins and errors associated for the energy resolution (bins per energy)
            e_bins, e_res_err = ctaplot.energy_resolution_per_energy(
                (np.array(energy_true.values) * u.TeV), (np.array(energy_pred) * u.TeV), bins=log_bins
            )
            
            # Calculate relative error and others errors bins for computation
            e = (e_bins[:-1].value + e_bins[1:].value) / 2
            e_res = [e_r[0] for e_r in e_res_err]
            e_res_minus = [e_r[0] - e_r[1] for e_r in e_res_err]
            e_res_plus = [e_r[2] - e_r[0] for e_r in e_res_err]
            e_res_min = [e_r[1] for e_r in e_res_err]
            e_res_max = [e_r[2] for e_r in e_res_err]
            
            # Add line to graph
            plt.errorbar(
                        e,
                        e_res,
                        yerr=[e_res_minus, e_res_plus],
                        label=f"Gamma",
                        markersize=8,
                        marker="o",
                        ls="--",
            )
            
            # Fill space between line points to show range
            plt.fill_between(e, e_res_min, e_res_max, alpha=0.2)
            
            # Compute graphic        
            plt.xscale("log")
            plt.xlabel("True Energy [TeV]")
            plt.ylabel("Energy Resolution [%]")
            plt.title("Energy Resolution", fontsize=20, fontweight=500)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")  
        else:
            print("No prediction Data found")


    def energy_bias(self):
        if not self.predictions is None :
            # Extract only wanted energies
            energy_pred = self.predictions["CTLearn_energy"]
            energy_true = self.predictions["true_energy"]
            
            # Build bins
            bins = (
                np.logspace(
                    np.log10(min(energy_true)),
                    np.log10(max(energy_true)),
                    num=int(
                        np.log10(max(energy_true) / min(energy_true))
                        * 5
                    )
                    + 1,
                )
            )
        
            # Calculate real error
            rel_err = (energy_pred - energy_true) / energy_true
        
            # Assign events to bins
            indices = np.digitize(energy_true, bins)
            results = []
        
            # For each bin
            for i in range(1, len(bins)):
                # Check if enough data in a bin
                mask = indices == i
                if np.sum(mask) < 20:
                    continue
        
                # Calculate mean bias (μ) and sigma gauss (σ) / Mean bias is the systematic offset between true and reconstructed energies / Sigma gauss is the standard deviation of relative error distribution
                err_bin = rel_err[mask]
                mu, sigma = norm.fit(err_bin)
        
                # Compute centers of each bin (logarithmic space)
                represent_energy = np.sqrt(bins[i-1] * bins[i])

                # Put results in an appropriate format for computation
                results.append({
                    "represent_energy": represent_energy,
                    "mean_bias": mu,
                    "sigma": sigma
                })

            # Store every bin results
            line_data = pd.DataFrame(results)
        
            # Add standard deviation to graph
            plt.plot(line_data["represent_energy"], line_data["sigma"], "o-", label="Standard deviation (Spread)")

            # Add bias to graph
            plt.plot(line_data["represent_energy"], line_data["mean_bias"], "o-", label="Bias")

            # Compute graphic
            plt.xscale("log")
            plt.xlabel("True Energy [TeV]")
            plt.ylabel("Bias / Std [TeV]")
            plt.title("Energy Bias and Standard deviation", fontsize=20, fontweight=500)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")  
        else:
            print("No prediction Data found")
            
    ###############################################################################
    ######################### DIRECTION REGRESSION GRAPHS #########################
    ###############################################################################

    def precision_in_degrees(self):
        if not self.predictions is None:
            # Extract predicated and true coordinates values
            alt_true = self.predictions["true_alt"]
            alt_pred = self.predictions["CTLearn_alt"]
            az_true = self.predictions["true_az"]
            az_pred = self.predictions["CTLearn_az"]

            # Get vector errors
            directional_errors = self.directional_error_deg(az_true, alt_true,
                                               az_pred, alt_pred)

            # Define degree threshold    
            thresholds = np.arange(1, 21)

            # Provide accuracy percentages for each threshold
            percentages = []
            total = len(directional_errors)
            for t in thresholds:
                p = np.sum(directional_errors <= t) / total * 100
                percentages.append(p)
            percentages = np.array(percentages)

            # Plot curve
            plt.figure(figsize=(8,5))
            plt.plot(thresholds, percentages)

            # Prepare graph
            plt.xlabel("Threshold [deg]")
            plt.ylabel("Percent within threshold [%]")
            plt.title("Degree Performance Curve (Camera)", fontsize=20, fontweight=500)
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.grid(True)
            plt.tight_layout()
            
            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8") 
        else:
            print("No prediction Data found")
            
    def angular_distribution(self):
        if not self.predictions is None:
            particles_type = self.predictions["particle_name"].unique()
            fig, axes = plt.subplots(1, len(particles_type), figsize=(4 * len(particles_type) , 3.5))
            # Loop for each particles
            for i, particle in enumerate(particles_type):
                # flatten if only one particle type
                if len(particles_type) > 1:
                    ax = axes[i]
                else:
                    ax = axes
            
                # get corresponding data    
                particle_df = self.predictions[self.predictions["particle_name"] == particle]
 
                # Provide a pointing area of the data individually
                ax.scatter(
                    particle_df["altitude"] / np.pi * 180,
                    particle_df["azimuth"] / np.pi * 180,
                    color="#cf004b",
                    label="Array pointing",
                    marker="x",
                    s=240,
                    edgecolor="#cf004b"
                )
              
                # Provide histogram of camera direction using degrees
                ax.hist2d(
                    particle_df["CTLearn_alt"],
                    particle_df["CTLearn_az"],
                    bins=100,
                    zorder=0,
                    cmap="viridis",
                    norm=plt.cm.colors.LogNorm(),
                )      
            
                # Finish graph visualisation for subplot
                ax.set_ylabel("Altitude [deg]")
                ax.set_xlabel("Azimuth [deg]")
                ax.legend()
                ax.set_title(f"{particle}")
                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label("Counts")

            # Prepare global graph
            fig.suptitle("Altitude-Azimuth (Camera) Distribution", fontsize=20, fontweight=500)
            plt.tight_layout()

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8") 
        else:
            print("No prediction Data found")

    def angular_resolution(self):
        if not self.predictions is None :
            # Extract predicated and true coordinates values (+ true energy for bin repartition)
            energy_true = self.predictions["true_energy"]
            alt_true = self.predictions["true_alt"]
            alt_pred = self.predictions["CTLearn_alt"]
            az_true = self.predictions["true_az"]
            az_pred = self.predictions["CTLearn_az"]
            
            
            # Build bins (0.1–100 TeV)
            log_bins = (
                np.logspace(
                    np.log10(min(energy_true)),
                    np.log10(max(energy_true)),
                    num=int(
                        np.log10(max(energy_true) / min(energy_true))
                        * 5
                    )
                    + 1,
                )
                * u.TeV
            )
            
            # Get bins and errors associated for the angular resolution (bins per energy)
            e_bins, ang_res_err = ctaplot.angular_resolution_per_energy(
                (np.array(alt_true.values) * u.deg),
                (np.array(alt_pred.values) * u.deg),
                (np.array(az_true.values) * u.deg),
                (np.array(az_pred.values) * u.deg),
                (np.array(energy_true.values) * u.TeV),
                bins=log_bins
            )

            # Calculate relative error and others errors bins for computation
            e = (e_bins[:-1].value + e_bins[1:].value) / 2
            ang_res = [e_r[0].value for e_r in ang_res_err]
            ang_res_minus = [e_r[0].value - e_r[1].value for e_r in ang_res_err]
            ang_res_plus = [e_r[2].value - e_r[0].value for e_r in ang_res_err]
            ang_res_min = [e_r[1].value for e_r in ang_res_err]
            ang_res_max = [e_r[2].value for e_r in ang_res_err]

            # Add line to graph
            plt.errorbar(e,
                        ang_res,
                        yerr=[ang_res_minus, ang_res_plus],
                        label="Gamma",
                        markersize=8,
                        marker="o",
                        ls="--",
            )
            
            # Fill space between line points to show range
            plt.fill_between(e, ang_res_min, ang_res_max, alpha=0.2)
            
            # Prepare Graph
            plt.xscale("log")
            plt.xlabel("True Energy [TeV]")
            plt.ylabel("Angular resolution [deg]")
            plt.title("Angular Resolution (Camera)", fontsize=20, fontweight=500)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")   
        else:
            print("No prediction Data found")

    def angular_bias(self):
        if not self.predictions is None :
            # Extract predicated and true coordinates values (+ true energy for bin repartition)
            energy_true = self.predictions["true_energy"]
            alt_true = self.predictions["true_alt"]
            alt_pred = self.predictions["CTLearn_alt"]
            az_true = self.predictions["true_az"]
            az_pred = self.predictions["CTLearn_az"]
                
            # Build bins
            bins = (
                np.logspace(
                    np.log10(min(energy_true)),
                    np.log10(max(energy_true)),
                    num=int(
                        np.log10(max(energy_true) / min(energy_true))
                        * 5
                    )
                    + 1,
                )
            )

            # Calculate linear and circular error
            alt_err = alt_pred - alt_true
            az_err = np.arctan2(np.sin(az_pred - az_true), np.cos(az_pred - az_true))
            
            
            # Assign events to bins
            indices = np.digitize(energy_true, bins)
            results_alt = []
            results_az = []
            
            # For each bin
            for i in range(1, len(bins)):
                # Check if enough data in a bin
                mask = indices == i
                if np.sum(mask) < 20:
                    continue
                    
                # Calculate mean bias (μ) and sigma gauss (σ) / Mean bias is the systematic offset between true and reconstructed energies / Sigma gauss is the standard deviation of relative error distribution
                alt_bin = alt_err[mask]
                mu_alt, sigma_alt = norm.fit(alt_bin)
            
                az_bin = az_err[mask]
                mu_az, sigma_az = norm.fit(az_bin)
            
                # Compute centers of each bin (logarithmic space)
                represent_energy = np.sqrt(bins[i-1] * bins[i])

                # Put Altitude results in a good format for computation
                results_alt.append({
                    "represent_energy": represent_energy,
                    "mean_bias": mu_alt,
                    "sigma": sigma_alt
                })
                # Put Azimuth results in a good format for computation
                results_az.append({
                    "represent_energy": represent_energy,
                    "mean_bias": mu_az,
                    "sigma": sigma_az
                })  
            # Store each bin formated in dataframes
            alt_line = pd.DataFrame(results_alt)
            az_line  = pd.DataFrame(results_az)
            
            # Add standard deviation to graph
            plt.figure(figsize=(8,5))
            plt.plot(alt_line["represent_energy"], alt_line["sigma"], "o-", label="Altitude - Standard Deviation")
            plt.plot(az_line["represent_energy"], az_line["sigma"], "x-", label="Azimuth- Standard Deviation")
            
            # Add bias to graph
            plt.plot(alt_line["represent_energy"], alt_line["mean_bias"], "o-", label="Altitude - Bias")
            plt.plot(az_line["represent_energy"], az_line["mean_bias"], "x-", label="Azimuth- Bias")
            
            # Compute graphic
            plt.xscale("log")
            plt.xlabel("True Energy [TeV]")
            plt.ylabel("Bias / Std [deg]")
            plt.title("Angular Bias and Standard deviation (Camera)", fontsize=20, fontweight=500)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            # Convert image to base64 (for rapid integration in html)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")   
        else:
            print("No prediction Data found")
            
    ##################################################################
    ######################### CUSTOM METRICS #########################
    ##################################################################

    # Apply correct class based on a threshold (normally probability computed)                
    def apply_corrected_predictions(self, row):
        if row["CTLearn_prediction"] >= self.threshold:
            return 1 # Gamma id
        else:
            return 0 # Proton (Hadron) id

    def coords_to_3D(self, az_deg, alt_deg):
        # Convert from degrees to radians
        az = np.radians(az_deg)
        alt = np.radians(alt_deg)
        # Get three dimensional values
        x = np.cos(alt) * np.cos(az)
        y = np.cos(alt) * np.sin(az)
        z = np.sin(alt)
        return np.stack([x, y, z], axis=1)

    def directional_error_deg(self, true_az, true_alt, pred_az, pred_alt):
        # Convert to 3D coords
        true_3d = self.coords_to_3D(true_az, true_alt)
        pred_3d = self.coords_to_3D(pred_az, pred_alt)
        # Dot product + clipping to prevent extreme
        dot = np.sum(true_3d * pred_3d, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        # Return converted back to degrees
        return np.degrees(np.arccos(dot))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a range of files to another SSH server.")
    parser.add_argument("--model_path", required=True, help="Path to the model information")
    parser.add_argument("--yaml_file", required=True, help="Yaml file for training information")
    parser.add_argument("--task", required=True, help="Type of task the model handles")
    parser.add_argument("--name", required=True, help="Name of the report")
    parser.add_argument("--output_path", required=True, help="Path where to save the report")

    args = parser.parse_args()
    
    report = Report(args.model_path, args.yaml_file, args.output_path, args.task, args.name)
