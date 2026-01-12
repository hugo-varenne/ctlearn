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
import math

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
from tabulate import tabulate
import matplotlib.ticker as ticker
from sklearn.calibration import calibration_curve

# CTLearn
import hdf5plugin, h5py
import ctaplot
import astropy.units as u
from traitlets.config import Config


class Report:
    def __init__(self, task_folder, output_path, task="type", name="Comparison of models"):
        
        self.task_folder = task_folder
        self.output_path = output_path
        self.task = task
        self.report_name = name
        self.metrics = []

        # Define the type of model
        match task:
            case "type":
                self.model_type = "Particle classification"
                self.threshold = 0.5
            case "energy":
                self.model_type = "Energy regression"
            case "cameradirection":
                self.model_type = "Camera Direction regression"
            case "skydirection":
                self.model_type = "Sky Direction regression"
            case _:
                self.model_type = "Unknow task"
                
        # Procedure to execute to build the report
        self.create_experiment_folder()
        self.load_models(self.task_folder)
        self.training_plot()
        self.get_prediction_data(task)

        # Depending on task, prepare different graphs
        if task == "type":
            self.get_type_graphs()
        elif task == "energy":
            self.get_energy_graphs()
        elif task == "cameradirection" or task == "skydirection":
            self.get_direction_graphs(task)
            
        # Rank and Extract metrics
        self.rank_model_metrics()
        self.rank_perf_metrics()
        self.build_report()
        
    def load_models(self, model_path):
        # Load models for a specific task
        self.model_list = {}
        for model in os.listdir(model_path):
            if model != "temp":
                model_file = os.path.join(model_path, model, "ctlearn_model.cpk")
                self.model_list[model] = tf.keras.models.load_model(model_file)

    def build_report(self):            
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
        <p class="metric"><strong>Number of Training Events :</strong> {self.training_events}</p>
        </div>
        <div class="col">
        <p><strong>Task :</strong> {self.model_type}</p>
        <p class="metric"><strong>Number of Testing Events :</strong> {self.testing_events}</p>
        <p><strong>Number of models implicated :</strong> {len(self.model_list)}</p>
        </div>
        </div>
        </section>
        
        <section>
        {self.model_metrics_template}
        </section>

        <section>
        {self.perf_metrics_template}
        </section>
        
        <section>
        <h2>Graphics</h2>
        <img src="data:image/png;base64,{self.training_plot}" alt="Training Loss Plot">
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
        plt.figure(figsize=(8,5))
        # Prepare colors
        colors = plt.cm.tab10.colors
        count = 1
        for model in os.listdir(self.task_folder):
            # Get training information
            training_results = os.path.join(self.task_folder, model, "training_log.csv")
            if os.path.exists(training_results):
                # Extract training results
                loss = pd.read_csv(training_results)
                # Get color for matching line
                color = colors[count % len(colors)]
                # Plot training and validation loss
                plt.plot(loss["epoch"], loss["loss"], linestyle="--", color=color, label="Training Loss")
                plt.plot(loss["epoch"], loss["val_loss"], color=color,  label="Validation Loss")
                count += 1
            else: 
                print(f"No training performed for the model {model}")

        # Prepare graphic
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

    ######################################################################        
    ######################### GET PREDICTED DATA #########################
    ###################################################################### 
    
    def get_prediction_data(self, task):
        self.predictions = {}
        for model in os.listdir(self.task_folder):
            # Get model predictions
            prediction_directory = os.path.join(self.task_folder, model, "predict")
            
            # Prepare DataFrame and mapping for classes
            model_predictions = pd.DataFrame()
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
                            model_predictions= pd.concat([model_predictions, combined_df], ignore_index=True)

            self.predictions[model] = model_predictions

    ###################################################################    
    ######################### METRICS DISPLAY #########################
    ###################################################################
    
    def rank_model_metrics(self):   
        ranking_metrics = ["num_params", "layers", "training_ms","inference_per_events", "estimated_flops"]
        rows = []
        for model in os.listdir(self.task_folder):
            metrics_path = os.path.join(self.task_folder, model, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as file:
                    metrics = json.load(file)
                if "inference_ms" in metrics and "training_ms" in metrics:
                    metrics["inference_per_events"] = metrics["inference_ms"]/metrics["testing_events"]
                metrics["model"] = model
                rows.append(metrics)
        self.model_metrics = pd.DataFrame(rows)

        float_cols = self.model_metrics.select_dtypes(include=["float"]).columns
        self.model_metrics[float_cols] = self.model_metrics[float_cols].applymap(
            lambda x: ('{:.3f}'.format(x)).rstrip('0').rstrip('.')
        )

        ranking_df = self.model_metrics[["model"]]
        for metric in ranking_metrics:
            if metric in self.model_metrics.columns:
                ranking_df[metric] = self.model_metrics[metric].rank(ascending=True, method="min", na_option='bottom').astype(int)
        
        ranking_df = ranking_df.set_index("model")

        # Catch number of events used
        training_events_list = self.model_metrics["training_events"]
        if training_events_list.nunique() == 1:
            self.training_events = training_events_list.iloc[0]
        else:
            self.training_events = f"approximatively {training_events_list.mean():.0f} (Not the same for each model)"
        
        testing_events_list = self.model_metrics["testing_events"]
        if testing_events_list.nunique() == 1:
            self.testing_events = testing_events_list.iloc[0]
        else:
            self.testing_events = f"approximatively {testing_events_list.mean():.0f} (Not the same for each model)"

        metrics_html = tabulate(
            self.model_metrics.drop(["training_events", "testing_events"], axis=1).set_index("model"),
            headers="keys",
            tablefmt="html",
            colalign=["center"] * len(self.model_metrics)
        )
        
        ranking_html = tabulate(
            ranking_df,
            headers="keys",
            tablefmt="html",
            colalign=["center"] * len(ranking_df)
        )
        
        self.model_metrics_template = f"""
        <div class="metrics-component">
        <style>
        .metrics-component table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }}
        .metrics-component th {{
            background: #2222224F;
            color: white;
            padding: 8px;
        }}
        .metrics-component td {{
            padding: 8px;
            border-bottom: 1px solid #ccc;
        }}
        /* rank colors */
        .metrics-component td.rank-1 {{ background-color: #c8f7c5; }}
        </style>
        
        <h2>Model and Runtime Summary</h2>
        {metrics_html}
        <h2>Ranking Summary</h2>
        {ranking_html}
        
        <script>
        // Color rank cells
        document.querySelectorAll(".metrics-component td").forEach(td => {{
            const v = td.innerText.trim();
            if (v === "1") td.classList.add("rank-1");
        }});
        </script>
        
        </div>
        """
    
    def rank_perf_metrics(self):
        if not self.predictions is None :
            if self.task == "energy":
                self.perf_metrics = pd.DataFrame(columns=["model", "MSE", "R²", "RMSLE", "MAE"])
                for model in self.predictions:
                    model_predictions = self.predictions[model]
                    predicted_values = model_predictions["CTLearn_energy"]
                    ground_truth = model_predictions["true_energy"]

                    ascending_metrics = {
                        "MSE": True,
                        "R²": False,
                        "RMSLE": True,
                        "MAE": True
                    }
                    
                    row = pd.DataFrame([{
                        "model": model,
                        "MSE": mean_squared_error(predicted_values, ground_truth),
                        "R²": r2_score(predicted_values, ground_truth),
                        "RMSLE": np.sqrt(tf.reduce_mean(np.square(ground_truth - predicted_values))),
                        "MAE": mean_absolute_error(predicted_values, ground_truth)
                    }])
                    self.perf_metrics = pd.concat([self.perf_metrics, row], ignore_index=True)        
            elif self.task == "type":
                self.perf_metrics = pd.DataFrame(columns=["model", "accuracy", "recall", "f1_score", "brier_score"])
                for model in self.predictions:
                    model_predictions = self.predictions[model]
                    labels = model_predictions["true_shower_primary_id"].unique()
                    predicted_values = model_predictions.apply(self.apply_corrected_predictions, axis=1)
                    ground_truth = model_predictions["particle_id"]

                    ascending_metrics = {
                        "accuracy": False,
                        "recall": False,
                        "f1_score": False,
                        "brier_score": True
                    }

                    row = pd.DataFrame([{
                        "model": model,
                        "accuracy": accuracy_score(ground_truth, predicted_values),
                        "recall": recall_score(ground_truth, predicted_values),
                        "f1_score": f1_score(ground_truth, predicted_values),
                        "brier_score": brier_score_loss(ground_truth, model_predictions["CTLearn_prediction"])
                    }])

                    self.perf_metrics = pd.concat([self.perf_metrics, row], ignore_index=True)
                    
            elif self.task == "cameradirection" or self.task == "skydirection":
                self.perf_metrics = pd.DataFrame(columns=["model", "mdae", "mae_circular", "rmsde", "mae_linear"])
                for model in self.predictions:
                    model_predictions = self.predictions[model]
                    # Extract necessary information to calculate metrics
                    alt_pred = model_predictions["CTLearn_alt"]
                    az_pred  = model_predictions["CTLearn_az"]
                    alt_true = model_predictions["true_alt"]
                    az_true  = model_predictions["true_az"]
                    # Calculate absolute difference
                    diff = np.abs(az_true - az_pred)
                    # Get angular error in degree
                    directional_errors = self.directional_error_deg(az_true, alt_true,
                                                   az_pred, alt_pred)
                    
                    ascending_metrics = {
                        "mdae": True,
                        "mae_circular": True,
                        "rmsde": True,
                        "mae_linear": True
                    }
                    
                    row = pd.DataFrame([{
                        "model": model,
                        "mdae": np.mean(directional_errors),
                        "mae_circular": np.mean(np.minimum(diff, 360-diff)),
                        "rmsde": np.sqrt(np.mean(directional_errors**2)),
                        "mae_linear": np.mean(np.abs(alt_true - alt_pred))
                    }])

                    self.perf_metrics = pd.concat([self.perf_metrics, row], ignore_index=True)
                    
            ranking_df = self.perf_metrics[["model"]]
            for metric in self.perf_metrics.drop("model", axis=1).columns:
                ranking_df[metric] = self.perf_metrics[metric].rank(ascending=ascending_metrics[metric], method="min", na_option='bottom').astype(int)
                    
            ranking_df = ranking_df.set_index("model")
            
            metrics_html = tabulate(
                self.perf_metrics.set_index("model"),
                headers="keys",
                tablefmt="html",
                colalign=["center"] * len(ranking_df),
                floatfmt=".3f"
            )
            
            ranking_html = tabulate(
                ranking_df,
                headers="keys",
                tablefmt="html",
                colalign=["center"] * len(ranking_df)
            )
            self.perf_metrics_template = f"""
            <div class="metrics-component">
            <style>
            .metrics-component table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }}
            .metrics-component th {{
                background: #2222224F;
                color: white;
                padding: 8px;
            }}
            .metrics-component td {{
                padding: 8px;
                border-bottom: 1px solid #ccc;
            }}
            /* rank colors */
            .metrics-component td.rank-1 {{ background-color: #c8f7c5; }}
            </style>
            
            <h2>Performance Summary</h2>
            {metrics_html}
            <h2>Ranking Summary</h2>
            {ranking_html}
            
            <script>
            // Color rank cells
            document.querySelectorAll(".metrics-component td").forEach(td => {{
                const v = td.innerText.trim();
                if (v === "1.0") td.classList.add("rank-1");
            }});
            </script>
            
            </div>
            """
        else:
            print("No prediction Data found")

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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 3.5 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]

                # Get predictions for specific model
                predictions = self.predictions[model]
                
                # Extract necessary information
                predicted_values = predictions.apply(self.apply_corrected_predictions, axis=1)
                ground_truth = predictions["particle_id"]
                labels = predictions["particle_name"].unique()
                
                 # Generate Confusion matrix
                cm = confusion_matrix(ground_truth, predicted_values)
                
                # Prepare graphic
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                disp.plot(ax=ax, cmap=plt.cm.Blues)
                ax.set_title(f'{model}')
                ax.set_ylabel("Ground truth")
                ax.set_xlabel("Predictions")
                ax.set_xticks(np.arange(-0.5, cm.shape[1], 0.5), minor=True)
                ax.set_yticks(np.arange(-0.5, cm.shape[0], 0.5), minor=True)
                ax.grid(which="minor", color="black", linewidth=0.5)
            
            # Visual fix if impair number of graphs    
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")
                
            # Prepare global graphic
            fig.suptitle("Confusion Matrix", fontsize=20, fontweight=500)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.45, top=0.8)
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
            for model in self.predictions:
                # Calculate ROC and AUC
                fpr_gamma, tpr_gamma, thresholds_gamma = metrics.roc_curve(self.predictions[model]["particle_id"], self.predictions[model]['CTLearn_prediction'], pos_label=1)
                auc_gamma = metrics.auc(fpr_gamma, tpr_gamma) 
                
                # Calculate ROC
                plt.plot(fpr_gamma, tpr_gamma, label=f"{model} (AUC={auc_gamma:.3f})")

            # Prepare graphic
            plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("CTLearn ROC Curve: Gamma vs Proton", fontsize=20, fontweight=500)
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
            for model in self.predictions:
                # Get predictions for corresponding model
                predictions = self.predictions[model]
                # Extract necessary information
                predicted_values = predictions["CTLearn_prediction"]
                ground_truth = predictions["particle_id"]
                
                # Get fractions of positives
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    ground_truth, predicted_values, n_bins=100, strategy="uniform"
                )
    
                # Display curve
                plt.plot(mean_predicted_value, fraction_of_positives, label=f"{model}")
                
            # Prepare graphic
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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get uniques particles
                predictions = self.predictions[model]
                particles_type = predictions["particle_name"].unique()
                
                # Display points for each particle type
                for particle in particles_type:
                    particle_data = predictions[predictions["particle_name"] == particle]
                    
                    # Plot histogram
                    ax.hist(
                        particle_data["CTLearn_prediction"],
                        bins=100,
                        range=(0, 1),
                        histtype="step",
                        density=True,
                        label=particle,
                    )
                # Show graphic    
                ax.set_title(f"{model}")
                ax.set_xlabel("Gammaness")
                ax.set_ylabel("Density")
                ax.legend()
            
            # Visual fix if impair number of graphs    
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")
                
            # Prepare global graphic
            fig.suptitle("Gammaness Distribution", fontsize=20, fontweight=500)
            plt.tight_layout()
            
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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get uniques particles
                predictions = self.predictions[model]
                
                # Get uniques particles
                particles_type = predictions["particle_name"].unique()
            
                # Prepare bins for graph
                log_bins = np.logspace(
                    np.log10(predictions["CTLearn_energy"].min()),
                    np.log10(predictions["CTLearn_energy"].max()),
                    100,
                )
                
                # Display points for each particle type
                for particle in particles_type:
                    particle_data = predictions[predictions["particle_name"] == particle]
            
                    # Build histogram
                    ax.hist(
                        particle_data["CTLearn_energy"],
                        bins=log_bins,
                        range=(0, 1),
                        histtype="step",
                        density=True,
                        label=particle,
                    )
                    
                # Show graphic
                ax.set_xlabel("Energy [TeV]")
                ax.set_ylabel("Density")
                ax.set_title(f"{model}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(title="Particle Types")
                
            # Visual fix if impair number of graphs
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")

            # Prepare global graphic
            fig.suptitle("Energy Distribution", fontsize=20, fontweight=500)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 3.5 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract only wanted energies
                particles_type = predictions["particle_name"].unique()
                if "Gamma" in particles_type:               
                    # Extract only wanted energies
                    particle_df = predictions[predictions["particle_name"] == "Gamma"]
                    
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
                    ax.set_title(f"{model}")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    
                    # Add a color bar to show repartition of values
                    cbar = plt.colorbar(ax.collections[0], ax=ax)
                    cbar.set_label("Number of events")
                else:
                    print("No gamma data found in predictions")

            # Visual fix if impair number of graphs
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")

            # Prepare global graphic
            fig.suptitle(f"Migration Matrix", fontsize=20, fontweight=500)
            plt.tight_layout()

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
            for model in self.predictions:
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract only wanted energies
                energy_pred = predictions["CTLearn_energy"]
                energy_true = predictions["true_energy"]
    
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
                            label=f"{model}",
                            markersize=8,
                            marker="o",
                            ls="--",
                        )
                
                # Fill space between line points to show range
                plt.fill_between(e, e_res_min, e_res_max, alpha=0.2)
                
            # Prepare graphic        
            plt.xscale("log")
            plt.xlabel("True Energy [TeV]")
            plt.ylabel("Energy Resolution [TeV]")
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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract only wanted energies
                energy_pred = predictions["CTLearn_energy"]
                energy_true = predictions["true_energy"]
                
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
                ax.plot(line_data["represent_energy"], line_data["sigma"], "o-", label="Standard deviation (Spread)")
    
                # Add bias to graph
                ax.plot(line_data["represent_energy"], line_data["mean_bias"], "o-", label="Bias")
    
                # Compute graphic
                ax.set_xscale("log")
                ax.set_xlabel("True Energy [TeV]")
                ax.set_ylabel("Bias / Std [TeV]")
                ax.set_title(f"{model}")
                ax.grid(True, which="both", ls="--", alpha=0.5)
                ax.legend()
                
            # Visual fix if impair number of graphs
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")

            # Prepare global graphic
            fig.suptitle(f"Energy Bias and Standard deviation", fontsize=20, fontweight=500)
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
            for model in self.predictions:
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract predicated and true coordinates values
                alt_true = predictions["true_alt"]
                alt_pred = predictions["CTLearn_alt"]
                az_true = predictions["true_az"]
                az_pred = predictions["CTLearn_az"]
    
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
                plt.plot(thresholds, percentages)

            # Prepare graph
            plt.xlabel("Threshold (degrees)")
            plt.ylabel("Percent within threshold (%)")
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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 3.5 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract only wanted energies
                particles_type = predictions["particle_name"].unique()
                if "Gamma" in particles_type:
                    # Extract only wanted energies
                    particle_df = predictions[predictions["particle_name"] == "Gamma"]
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
                    ax.set_title(f"{model}")
                    
                    # Add a color bar to show repartition of values
                    cbar = plt.colorbar(ax.collections[0], ax=ax)
                    cbar.set_label("Number of events")
                else:
                    print("No gamma data found in predictions")

            # Visual fix if impair number of graphs
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")
            
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
            for model in self.predictions:
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract predicated and true coordinates values (+ true energy for bin repartition)
                energy_true = predictions["true_energy"]
                alt_true = predictions["true_alt"]
                alt_pred = predictions["CTLearn_alt"]
                az_true = predictions["true_az"]
                az_pred = predictions["CTLearn_az"]
                
                
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
                            label=f"{model}",
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
            plt.tight_layout()

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
            # Prepare subplots layout
            rows = math.ceil(len(self.predictions) / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
            axes = axes.flatten()
            for index, model in enumerate(self.predictions):
                ax = axes[index]
                # Get corresponding model predictions
                predictions = self.predictions[model]
                
                # Extract predicated and true coordinates values (+ true energy for bin repartition)
                energy_true = predictions["true_energy"]
                alt_true = predictions["true_alt"]
                alt_pred = predictions["CTLearn_alt"]
                az_true = predictions["true_az"]
                az_pred = predictions["CTLearn_az"]
                    
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
                ax.plot(alt_line["represent_energy"], alt_line["sigma"], "o-", label="Altitude - Standard Deviation")
                ax.plot(az_line["represent_energy"], az_line["sigma"], "x-", label="Azimuth- Standard Deviation")
                
                # Add bias to graph
                ax.plot(alt_line["represent_energy"], alt_line["mean_bias"], "o-", label="Altitude - Bias")
                ax.plot(az_line["represent_energy"], az_line["mean_bias"], "x-", label="Azimuth- Bias")
                
                # Compute graphic
                ax.set_xscale("log")
                ax.set_xlabel("True Energy [TeV]")
                ax.set_ylabel("Bias / Std [def]")
                ax.set_title(f"{model}")
                ax.grid(True, which="both", ls="--", alpha=0.5)
                ax.legend()
                
            # Visual fix if impair number of graphs
            for j in range(index + 1, len(axes)):
                axes[j].axis("off")

            # Prepare global graphic
            fig.suptitle(f"Angular Bias and Standard deviation (Camera)", fontsize=20, fontweight=500)
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
    parser.add_argument("--task_folder", required=True, help="Path to the task folder containing information about models")
    parser.add_argument("--task", required=True, help="Task for the model")
    parser.add_argument("--name", required=True, help="Name of the report")
    parser.add_argument("--output_path", required=True, help="Path where to save the report")

    args = parser.parse_args()
    
    report = Report(args.task_folder, args.output_path, args.task, args.name)
