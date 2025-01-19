import os
import json
import pickle
from tensorflow.keras.models import load_model, save_model

from ..core.element_manager import ElementManager

class ModelStorage:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def save_state(self, model, model_name, state_data, element_encoder):
        """Save model and state data"""
        base_path = os.path.join(self.data_path, model_name)
        os.makedirs(base_path, exist_ok=True)
        
        # Save model
        model.save(os.path.join(base_path, "model.keras"))
        
        # Save state data
        with open(os.path.join(base_path, "data.json"), 'w') as f:
            json.dump(state_data, f, indent=4)
            
        with open(os.path.join(base_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(element_encoder, f)
            
        print(f"Model saved to {base_path}")
            
    def load_state(self, model_name):
        """Load state data"""
        base_path = os.path.join(self.data_path, model_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model folder not found: {base_path}")
            
        # Load state data
        with open(os.path.join(base_path, "data.json"), 'r') as f:
            state = json.load(f)
        
        instance = ElementManager()
        
        instance.base_elements = state["base_elements"]
        instance.combinations = {
            (tuple(combo), tuple(result)) for [combo, result] in state['combinations']
        }
        
        # Load invalid combinations
        instance.invalid_combinations = {
            tuple(combo) for combo in state.get('invalid_combinations', [])
        }
        
        instance.all_elements = state['all_elements']
        instance.element_emojis = state.get('element_emojis', dict(instance.base_elements))
        
        with open(os.path.join(base_path, "encoder.pkl"), 'rb') as f:
            instance.element_encoder = pickle.load(f)
            
        return instance
            
    def load_model(self, model_name):
        """Load saved model"""
        model_path = os.path.join(self.data_path, model_name, "model.keras")
        return load_model(model_path)
        
    def list_available_models(self):
        """List all available models"""
        if not os.path.exists(self.data_path):
            return []
        
        models = []
        for model_name in os.listdir(self.data_path):
            model_path = os.path.join(self.data_path, model_name)
            if os.path.isdir(model_path) and all(
                os.path.exists(os.path.join(model_path, f))
                for f in ["model.keras", "data.json"]
            ):
                models.append(model_name)
        return models