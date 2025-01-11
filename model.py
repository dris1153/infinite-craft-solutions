import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class ElementCombiner:
    def __init__(self):
        # Define base elements
        self.base_elements = ['fire', 'water', 'wind', 'earth']
        
        # Define known combinations and results
        self.combinations = [
            (['water', 'fire'], 'steam'),
            (['water', 'earth'], 'plant'),
            (['fire', 'steam'], 'engine'),
            (['engine', 'earth'], 'tractor')
        ]
        
        self._initialize_elements()
    
    def _initialize_elements(self):
        # Create a set of all unique elements
        self.all_elements = set(self.base_elements)
        for combo, result in self.combinations:
            self.all_elements.add(result)
            self.all_elements.update(combo)
        
        self.all_elements = list(self.all_elements)
        
        # Initialize label encoders
        self.element_encoder = LabelEncoder()
        self.element_encoder.fit(self.all_elements)

    def save_state(self, model, model_name):
        """Save model, combinations, and encoder state"""
        # Create directories if they don't exist
        base_path = os.path.join("datas", model_name)
        os.makedirs(base_path, exist_ok=True)
        
        # Save the model
        # save_model(model, os.path.join(base_path, "model.h5"))

        model.save(os.path.join(base_path, "model.keras"))
        
        # Save combinations and elements
        state = {
            'base_elements': self.base_elements,
            'combinations': self.combinations,
            'all_elements': self.all_elements
        }
        
        with open(os.path.join(base_path, "data.json"), 'w') as f:
            json.dump(state, f, indent=4)
            
        # Save the LabelEncoder
        with open(os.path.join(base_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(self.element_encoder, f)
            
        print(f"Model saved in folder: {base_path}")
            
    @classmethod
    def load_state(cls, model_name):
        """Load a saved state and return a new ElementCombiner instance"""
        base_path = os.path.join("datas", model_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model folder not found: {base_path}")
            
        instance = cls.__new__(cls)
        
        # Load combinations and elements
        with open(os.path.join(base_path, "data.json"), 'r') as f:
            state = json.load(f)
            
        instance.base_elements = state['base_elements']
        instance.combinations = state['combinations']
        instance.all_elements = state['all_elements']
        
        # Load the LabelEncoder
        with open(os.path.join(base_path, "encoder.pkl"), 'rb') as f:
            instance.element_encoder = pickle.load(f)
            
        return instance
    
    @staticmethod
    def load_model(model_name):
        """Load a saved model"""
        model_path = os.path.join("datas", model_name, "model.keras")
        return load_model(model_path)
    
    @staticmethod
    def list_available_models():
        """List all available models in the datas directory"""
        if not os.path.exists("datas"):
            return []
        
        models = []
        for model_name in os.listdir("datas"):
            model_path = os.path.join("datas", model_name)
            if os.path.isdir(model_path) and all(
                os.path.exists(os.path.join(model_path, f))
                for f in ["model.keras", "data.json", "encoder.pkl"]
            ):
                models.append(model_name)
        return models
        
    def add_combination(self, elem1, elem2, result):
        """Add a new combination to the training data"""
        self.combinations.append(([elem1, elem2], result))
        
        # Update all_elements if new elements were introduced
        new_elements = {elem1, elem2, result}
        if not new_elements.issubset(self.all_elements):
            self.all_elements.extend(list(new_elements - set(self.all_elements)))
            self.element_encoder.fit(self.all_elements)
        
    def prepare_data(self):
        X = []  # Input combinations
        y = []  # Results
        
        # Convert combinations to training data
        for (elem1, elem2), result in self.combinations:
            # Encode input elements
            elem1_encoded = self.element_encoder.transform([elem1])[0]
            elem2_encoded = self.element_encoder.transform([elem2])[0]
            
            # Create input vector
            input_vector = np.zeros(len(self.all_elements) * 2)
            input_vector[elem1_encoded] = 1
            input_vector[elem2_encoded + len(self.all_elements)] = 1
            
            X.append(input_vector)
            
            # Encode output
            result_encoded = self.element_encoder.transform([result])[0]
            y.append(result_encoded)
        
        X = np.array(X)
        y = to_categorical(y, num_classes=len(self.all_elements))
        
        return X, y
    
    def create_model(self):
        input_dim = len(self.all_elements) * 2
        output_dim = len(self.all_elements)
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train_model(self, model, epochs=200):
        X, y = self.prepare_data()
        model.fit(X, y, epochs=epochs, verbose=0)
        
    def predict_combination(self, model, element1, element2):
        # Encode input elements
        try:
            elem1_encoded = self.element_encoder.transform([element1])[0]
            elem2_encoded = self.element_encoder.transform([element2])[0]
        except ValueError:
            return "Unknown element(s)", 0.0
        
        # Create input vector
        input_vector = np.zeros(len(self.all_elements) * 2)
        input_vector[elem1_encoded] = 1
        input_vector[elem2_encoded + len(self.all_elements)] = 1
        
        # Make prediction
        prediction = model.predict(np.array([input_vector]), verbose=0)
        predicted_idx = np.argmax(prediction[0])
        
        # Decode prediction
        result = self.element_encoder.inverse_transform([predicted_idx])[0]
        confidence = prediction[0][predicted_idx]
        
        return result, confidence

def main():
    print("Choose an option:")
    print("1. Start new model")
    print("2. Load existing model")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "2":
        # Show available models
        available_models = ElementCombiner.list_available_models()
        if not available_models:
            print("No saved models found. Starting new model...")
            combiner = ElementCombiner()
            model = combiner.create_model()
            combiner.train_model(model)
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            model_choice = input("\nEnter model number or name: ")
            try:
                # Try to get model by number
                idx = int(model_choice) - 1
                if 0 <= idx < len(available_models):
                    model_name = available_models[idx]
                else:
                    model_name = model_choice
            except ValueError:
                model_name = model_choice
            
            try:
                combiner = ElementCombiner.load_state(model_name)
                model = ElementCombiner.load_model(model_name)
                print(f"Model '{model_name}' loaded successfully!")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Starting new model...")
                combiner = ElementCombiner()
                model = combiner.create_model()
                combiner.train_model(model)
    else:
        # Start new model
        combiner = ElementCombiner()
        model = combiner.create_model()
        combiner.train_model(model)
    
    while True:
        print("\nEnter two elements to combine (or 'quit' to exit, 'save' to save model):")
        elem1 = input("First element: ").lower()
        
        if elem1 == 'quit':
            break
        elif elem1 == 'save':
            model_name = input("Enter model name: ")
            combiner.save_state(model, model_name)
            continue
        
        elem2 = input("Second element: ").lower()
        if elem2 == 'quit':
            break
        
        # Make prediction
        result, confidence = combiner.predict_combination(model, elem1, elem2)
        print(f"\nPredicted result: {result}")
        print(f"Confidence: {confidence:.2%}")
        
        # Ask for evaluation
        evaluation = input("\nIs this prediction correct? (yes/no): ").lower()
        
        if evaluation == 'no':
            # Get correct answer and retrain
            correct_result = input("What is the correct result? ").lower()
            
            # Add new combination to training data
            combiner.add_combination(elem1, elem2, correct_result)
            
            # Recreate and retrain model with updated data
            model = combiner.create_model()
            combiner.train_model(model)
            print("\nModel retrained with new combination!")
        
        print("\nKnown combinations:")
        for (e1, e2), res in combiner.combinations:
            print(f"{e1} + {e2} = {res}")

if __name__ == "__main__":
    main()