# New classes and external libraries can be used to enhance the current architecture with two separated modules (backbone and head)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Backbone model (always used) and can be only element called if overwrite of head is wanted
# Parameters can be changed or added (name MUST end with "_block")
class ModelBackbone(Model):
    def __init__(self, dropout_rate=0.2, name="backbone_block"):
        super().__init__(name=name)

        # Define layers

        
        # Need to flatten before head
    def call(self, inputs, training=False):
        
        # Build the model by chaining different layers
        return ...

        
# Head model (can be overwritten) / Need to converge for a specific task
# Parameters can be changed or added (reco_task can only be between ["type", "cameradirection", "skydirection", "energy"])
class ModelHead(Model):
    def __init__(self, num_classes=10, reco_task="type", dropout_rate=0.2, name="task_name"):
        super().__init__(name=name)
        
        # Define layers (can be overwritten)
        
        if reco_task == "type":
            # ouput layer in case of type task
        else:
            # ouput layer in case of type task

    
    def call(self, inputs, training=False):
        # Build the model by chaining different layers
        return ...

# Complex Model as a Class (Global class combining head and backbone)
class ComplexModel(Model):
    def __init__(self, input_shape=(96, 96, 2), reco_task="type", num_classes=10, dropout_rate=0.2, name="complex_test_block"):
        super().__init__(name=name)

        # Initialize model parameters
        self.input_dim = input_shape
        self.task = reco_task
        self.num_classes = num_classes 
        self.model_name = name
        self.dropout_rate=dropout_rate
        
        # Build backbone and head
        self.backbone = self.build_backbone()
        self.head = self.build_head(name=self.task) # Backbone must be initialized before the head as parameters from the backbone are required

        # Prepare and build global model for graphic usage
        inp = Input(shape=self.input_dim, name="input")
        x = self.backbone(inp)
        out = self.head(x)
        self._graph_model = Model(inp, out, name=f"{name}_functional")

    # Build by chaining different layers
    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        out = self.head(x, training=training)
        return out
        
    # Build backbone as functional layer (combine backbone layers as one for tensorflow recognition) / name NEED to end with _block if provided
    def build_backbone(self, name="backbone_block"):
        inp = Input(shape=self.input_dim)
        # Create a backbone block
        backbone = ModelBackbone(dropout_rate=self.dropout_rate, name=name)
        out = backbone(inp)
        return Model(inputs=inp, outputs=out, name=name)        

    # Build head as functional layer (combine head layers as one for tensorflow recognition)   
    def build_head(self, name="task"):
        inp = Input(shape=self.backbone.output_shape[1:])
        # Create a head block
        head = ModelHead(num_classes=self.num_classes, reco_task=self.task, dropout_rate=self.dropout_rate, name=name)
        out = head(inp)
        return Model(inputs=inp, outputs=out, name=name)
        
    # Override get_config to ensure custom serialization
    def get_config(self):
        return {
            "name": self.name,
            "backbone": self.backbone.name,
            "head": self.head.name,
        }

    # Add save/load helpers for the functional graph
    def save(self, path, **kwargs):
        """Save using the internal functional graph"""
        self._graph_model.save(path, **kwargs)

    @classmethod
    def load(cls, path):
        """Reload both structure and callable backbone/head"""
        graph_model = tf.keras.models.load_model(path, compile=False)
        model = cls(input_shape=graph_model.input_shape[1:])
        model._graph_model = graph_model
        return model
