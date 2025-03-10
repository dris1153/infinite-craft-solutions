# Standard library imports
import os
import json
import pickle
import threading
import random
import time

# Data processing and scientific computing
import numpy as np

# Deep learning framework
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# Machine learning utilities
from sklearn.preprocessing import LabelEncoder

# HTTP requests
import requests


epochs_config = 100
batch_size_config = 64

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datas"))

class ElementCombiner:
    def __init__(self):
        # Define base elements with emojis
        self.base_elements = {
            'fire': '🔥',
            'water': '💧',
            'wind': '💨',
            'earth': '🌎'
        }
        
        self.invalid_combinations = set()
        
        # Store combinations as sorted tuples for different elements
        self.combinations = {
            (tuple(sorted(['water', 'fire'])), ('steam', '💨')),
            (tuple(sorted(['water', 'earth'])), ('plant', '🌱')),
            (tuple(sorted(['fire', 'earth'])), ('lava', '🌋')),
            (tuple(sorted(['wind', 'water'])), ('wave', '🌊')),
            (tuple(sorted(['wind', 'fire'])), ('smoke', '💨')),
            (tuple(sorted(['wind', 'earth'])), ('dust', '🌫️')),
            # Same-element combinations (no need to sort)
            ((tuple(['fire', 'fire'])), ('inferno', '🔥')),
            ((tuple(['water', 'water'])), ('ocean', '🌊')),
            ((tuple(['wind', 'wind'])), ('tornado', '🌪️')),
            ((tuple(['earth', 'earth'])), ('mountain', '⛰️'))
        }
        
        self._initialize_elements()
    
    def _initialize_elements(self):
        """Initialize elements and encoders"""
        # Create a set of all unique elements
        self.all_elements = set(self.base_elements.keys())
        self.element_emojis = dict(self.base_elements)
        
        for combo, result in self.combinations:
            result_name, result_emoji = result
            self.all_elements.add(result_name)
            self.all_elements.update(combo)
            self.element_emojis[result_name] = result_emoji
        
        # Convert to sorted list for consistent encoding
        self.all_elements = list(sorted(self.all_elements))
        
        # Initialize label encoder
        self.element_encoder = LabelEncoder()
        self.element_encoder.fit(self.all_elements)
        
    def add_invalid_combination(self, elem1, elem2):
        """Add a combination that's known to be invalid"""
        combo_key = self.get_combination_key(elem1, elem2)
        self.invalid_combinations.add(combo_key)

    def is_invalid_combination(self, elem1, elem2):
        """Check if a combination is known to be invalid"""
        combo_key = self.get_combination_key(elem1, elem2)
        return combo_key in self.invalid_combinations
 
    def get_api_combination(self, elem1, elem2):
        """Get combination result from the API"""
        try:
            response = requests.get(f'https://infiniteback.org/pair?first={elem1.capitalize()}&second={elem2.capitalize()}')
            if response.status_code == 200:
                data = response.json()
                print("+++",data,"+++")
                if data is not None:
                    return data['result'].lower(), data['emoji']
                else:
                    self.add_invalid_combination(elem1, elem2)
                    return None, None
            return None, None
        except Exception as e:
            print(f"API Error: {e}")
            return None, None

    def predict_combination(self, model, element1, element2):
        """Predict the result of combining two elements"""
        # First check if this is a known invalid combination
        if self.is_invalid_combination(element1, element2):
            return None, None, 1.0
        # First check if this is a known combination
        known_result = self.get_combination_result(element1, element2)
        if known_result is not None:
            return known_result[0], known_result[1], 1.0
        
        # For new combinations, call the API
        new_element, emoji = self.get_api_combination(element1, element2)
        if new_element and emoji:
            return new_element, emoji, 0.5
        elif new_element is None:  # API indicated invalid combination
            return None, None, 0.5
        
        # Fallback to model prediction if API fails
        return None, None, 0.0

    def get_combination_key(self, elem1, elem2):
        """Get the standardized combination key"""
        if elem1 == elem2:
            return tuple([elem1, elem2])  # Keep same-element combinations as-is
        return tuple(sorted([elem1, elem2]))  # Sort different elements

    def get_combination_result(self, elem1, elem2):
        """Get the known result for a combination of elements"""
        combo_key = self.get_combination_key(elem1, elem2)
        for combo, result in self.combinations:
            if combo == combo_key:  # Now we can compare directly
                return result
        return None
    
    def add_combination(self, elem1, elem2, result, emoji):
        """Add a new combination to the training data"""
        combo_key = self.get_combination_key(elem1, elem2)
        
        # Remove any existing combination with these elements
        self.combinations = {c for c in self.combinations if tuple(c[0]) != combo_key}
        
        # Add the new combination
        self.combinations.add((combo_key, (result, emoji)))
        
        # Update all_elements and emojis if new elements were introduced
        if result not in self.all_elements:
            self.all_elements.append(result)
            self.element_emojis[result] = emoji
            self.element_encoder.fit(self.all_elements)

    def prepare_data(self, batch_size=batch_size_config):
        """Prepare training data for the model including invalid combinations with batch support"""
        X = []  # Input combinations
        y = []  # Results
        
        print("*** Prepare process: 0% ***")
        print("-> Get length of all elements")
        # Get the total number of classes including valid elements and invalid combination class
        num_classes = len(self.all_elements)
        
        print("*** Prepare process: 10% ***")
        print("-> Convert valid combinations to training data")
        
        combinations_len = len(self.combinations)
        combinations_index = 0
        # Convert valid combinations to training data
        for combo, (result, _) in self.combinations:
            combinations_index += 1
            
            if combinations_index % 1000 == 0:
                print(f"*** Prepare process: {10 + int((combinations_index / combinations_len) * 40)}% ***")
                print(f"-> Processing combination {combinations_index}/{combinations_len}")
            
            elem1, elem2 = tuple(combo)
            
            # Encode input elements
            elem1_encoded = self.element_encoder.transform([elem1])[0]
            elem2_encoded = self.element_encoder.transform([elem2])[0]
            
            # Create input vector
            input_vector = np.zeros(len(self.all_elements) * 2)
            input_vector[elem1_encoded] = 1
            input_vector[elem2_encoded + len(self.all_elements)] = 1
            
            # For valid combinations, use the actual result class
            result_encoded = self.element_encoder.transform([result])[0]
            if result_encoded >= num_classes:
                continue
            
            X.append(input_vector)
            y.append(result_encoded)
            
            # Add reversed combination if elements are different
            if elem1 != elem2:
                input_vector_rev = np.zeros(len(self.all_elements) * 2)
                input_vector_rev[elem2_encoded] = 1
                input_vector_rev[elem1_encoded + len(self.all_elements)] = 1
                X.append(input_vector_rev)
                y.append(result_encoded)

        print("*** Prepare process: 50% ***")
        print("-> Convert invalid combinations to training data")
        
        invalid_combinations_len = len(self.invalid_combinations)
        invalid_combinations_index = 0
        
        # Add invalid combinations to training data
        for combo in self.invalid_combinations:
            invalid_combinations_index += 1
            
            if invalid_combinations_index % 10000 == 0:
                print(f"*** Prepare process: {50 + int((invalid_combinations_index / invalid_combinations_len) * 40)}% ***")
                print(f"-> Processing invalid combination {invalid_combinations_index}/{invalid_combinations_len}")
            
            elem1, elem2 = tuple(combo)
            
            try:
                # Encode input elements
                elem1_encoded = self.element_encoder.transform([elem1])[0]
                elem2_encoded = self.element_encoder.transform([elem2])[0]
                
                # Create input vector
                input_vector = np.zeros(len(self.all_elements) * 2)
                input_vector[elem1_encoded] = 1
                input_vector[elem2_encoded + len(self.all_elements)] = 1
                
                # For invalid combinations, use the last class index
                invalid_class = num_classes - 1
                
                X.append(input_vector)
                y.append(invalid_class)
                
                # Add reversed combination if elements are different
                if elem1 != elem2:
                    input_vector_rev = np.zeros(len(self.all_elements) * 2)
                    input_vector_rev[elem2_encoded] = 1
                    input_vector_rev[elem1_encoded + len(self.all_elements)] = 1
                    X.append(input_vector_rev)
                    y.append(invalid_class)
            except Exception as e:
                print(f"Error processing invalid combination {elem1} + {elem2}: {e}")
                continue
        
        if not X:  # Check if we have any valid training data
            raise ValueError("No valid training data could be generated")
        
        print("*** Prepare process: 90% ***")
        print("-> Convert data to NumPy arrays")
        
        X = np.array(X)
        y = np.array(y)
        
        # Verify that all labels are within valid range
        if np.any(y >= num_classes):
            raise ValueError(f"Found label(s) >= number of classes ({num_classes})")
        
        print("*** Prepare process: 98% ***")
        print("-> Convert labels to one-hot encoding")
        
        y = to_categorical(y, num_classes=num_classes)
        
        # Create TensorFlow dataset with batching
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print("*** Prepare process: 100% ***")
        print("-> Data preparation complete!")
        return dataset
    
    def create_model(self):
        """Create and return a new neural network model with invalid combination support"""
        input_dim = len(self.all_elements) * 2
        output_dim = len(self.all_elements)  # Number of valid elements
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model

    
    def train_model(self, model, epochs=epochs_config, batch_size=batch_size_config):
        """Train the model on the current combinations with progress tracking"""
        print("******************************************")
        print("Preparing data for training...")
        dataset = self.prepare_data(batch_size)
        
        # Create a custom callback to track progress
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                self.last_update_time = time.time()
                self.update_interval = 1  # Update every 1 second
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                
            def on_epoch_end(self, epoch, logs=None):
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    progress = (epoch + 1) / self.total_epochs * 100
                    elapsed_time = current_time - self.epoch_start_time
                    eta = elapsed_time * (self.total_epochs - epoch - 1)
                    
                    print(f"\rTraining Progress: {progress:.1f}% "
                        f"[Epoch {epoch + 1}/{self.total_epochs}] "
                        f"Loss: {logs.get('loss', 0):.4f} "
                        f"Accuracy: {logs.get('accuracy', 0):.4f} "
                        f"ETA: {eta:.1f}s", end="")
                    
                    self.last_update_time = current_time
        
        print("\nModel fitting...")
        # Initialize our custom callback
        progress_callback = TrainingProgressCallback(epochs)
        
        # Train the model with the callback
        history = model.fit(
            dataset,
            epochs=epochs,
            verbose=0,
            callbacks=[progress_callback]
        )
        
        print("\nTraining complete!")
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
        print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
        print("******************************************")
        
        return history

    def get_train_elements(self, uncombined, index):
        # Prioritize uncombined pairs
        chosen_pair = uncombined[index]
        return chosen_pair[0], chosen_pair[1]

    def auto_train(self, model, amount, auto_pass=False):
        """Auto train the model with random combinations"""
        trained_count = 0
        
        print(f"\nStarting auto training for {amount} combinations...")
        print("Press Ctrl+C to stop at any time\n")
        
        """Get two random elements, prioritizing uncombined pairs"""
        all_elements_list = list(self.all_elements)
        
        # Create a set of all possible combinations
        all_possible = set()
        for i, elem1 in enumerate(all_elements_list):
            # Include same-element combinations
            if elem1 != "":
                all_possible.add(tuple([elem1, elem1]))
            # Include combinations with other elements
            for elem2 in all_elements_list[i:]:  # Start from i to avoid duplicates
                if elem1 != "" and elem2 != "" and elem1 != elem2:
                    all_possible.add(tuple(sorted([elem1, elem2])))
        
        print("-------------------------------------")
        print(f"Total possible combinations: {len(all_possible)}")
        
        # Get set of existing combinations
        existing_combos = {tuple(combo) for combo, _ in self.combinations}
        
        # Get set of invalid combinations
        invalid_combos = {tuple(combo) for combo in self.invalid_combinations}
        
        # Find combinations that don't exist yet
        uncombined = list(all_possible - existing_combos - invalid_combos)
        
        try:
            while trained_count < amount:
                
                print("-------------------------------------")
                
                elem1, elem2 = self.get_train_elements(uncombined, trained_count)
                
                print(f"\nTrying combination {trained_count + 1}/{amount}")
                print(f"Elements: {elem1} {self.element_emojis.get(elem1, '')} + "
                    f"{elem2} {self.element_emojis.get(elem2, '')}")
                
                result, emoji, confidence = self.predict_combination(model, elem1, elem2)
                
                if result:
                    print(f"Predicted result: {result} {emoji}")
                    print(f"Confidence: {confidence:.2%}")
                    
                    if auto_pass:
                        if confidence < 1.0:
                            self.add_combination(elem1, elem2, result, emoji)
                            print("Combination automatically added!")
                            trained_count += 1
                    else:
                        agree_result = input("\nDo you agree with this result? (yes/no/skip): ").lower()
                        
                        if agree_result == 'skip':
                            continue
                        elif agree_result == 'no':
                            correct_result = input("What is the correct result? ").lower()
                            correct_emoji = input("What emoji should represent this result? ")
                            
                            self.add_combination(elem1, elem2, correct_result, correct_emoji)
                            print("\nCombination added to known combinations!")
                            trained_count += 1
                        elif agree_result == 'yes' and confidence < 1.0:
                            self.add_combination(elem1, elem2, result, emoji)
                            print("\nSuggested combination added to known combinations!")
                            trained_count += 1
                else:
                    print("\nFailed to generate combination. Trying another...")
                    trained_count += 1
                    
                print(f"\nProgress: {trained_count}/{amount} combinations trained")
            model = self.create_model()
            self.train_model(model)  
        except KeyboardInterrupt:
            print("\n\nAuto training interrupted by user")
        
        return model
    
    def save_state(self, model, model_name):
        """Save model, combinations, invalid combinations, and encoder state"""
        base_path = os.path.join(data_path, model_name)
        os.makedirs(base_path, exist_ok=True)
        
        model.save(os.path.join(base_path, "model.keras"))
        
        # Convert tuple to lists for JSON serialization
        serializable_combinations = [
            ([list(combo), list(result)]) for combo, result in self.combinations
        ]
        
        # Convert invalid combinations to list of lists for JSON
        serializable_invalid = [list(combo) for combo in self.invalid_combinations]
        
        state = {
            'base_elements': self.base_elements,
            'combinations': serializable_combinations,
            'invalid_combinations': serializable_invalid,
            'all_elements': self.all_elements,
            'element_emojis': self.element_emojis
        }
        
        with open(os.path.join(base_path, "data.json"), 'w') as f:
            json.dump(state, f, indent=4)
            
        with open(os.path.join(base_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(self.element_encoder, f)
            
        print(f"Model saved in folder: {base_path}")
          
    # Add this new method to the ElementCombiner class
    def get_stats(self):
        """Get statistics about the current model state"""
        stats = {
            'total_elements': len(self.all_elements),
            'base_elements': len(self.base_elements),
            'derived_elements': len(self.all_elements) - len(self.base_elements),
            'total_combinations': len(self.combinations),
            'total_invalid_combinations': len(self.invalid_combinations),
            'most_versatile_elements': self._get_most_versatile_elements(),
        }
        return stats

    def _get_most_versatile_elements(self):
        """Find elements that appear in the most combinations"""
        element_counts = {}
        for combo, _ in self.combinations:
            for elem in combo:
                element_counts[elem] = element_counts.get(elem, 0) + 1
        
        # Get top 3 most used elements
        sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
        return [(elem, count) for elem, count in sorted_elements[:3]]

    @classmethod
    def load_state(cls, model_name):
        """Load a saved state"""
        base_path = os.path.join(data_path, model_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model folder not found: {base_path}")
            
        instance = cls.__new__(cls)
        
        # Load state data
        with open(os.path.join(base_path, "data.json"), 'r') as f:
            state = json.load(f)
            
        instance.base_elements = state['base_elements']
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
    
    @staticmethod
    def load_model(model_name):
        """Load a saved model"""
        model_path = os.path.join(data_path, model_name, "model.keras")
        return load_model(model_path)
    
    @staticmethod
    def list_available_models():
        """List all available models in the datas directory"""
        if not os.path.exists(data_path):
            return []
        
        models = []
        for model_name in os.listdir(data_path):
            model_path = os.path.join(data_path, model_name)
            if os.path.isdir(model_path) and all(
                os.path.exists(os.path.join(model_path, f))
                for f in ["model.keras", "data.json", "encoder.pkl"]
            ):
                models.append(model_name)
        return models

    def generate_js_code(combiner):
        """Generate JavaScript code for localStorage"""
        # Convert elements to required format
        elements = []
        
        # Add base elements
        for text, emoji in combiner.base_elements.items():
            elements.append({
                "text": text.capitalize(),
                "emoji": emoji,
                "discovered": False
            })
        
        # Add combined elements
        for combo, (text, emoji) in combiner.combinations:
            if text not in combiner.base_elements:  # Avoid duplicates
                elements.append({
                    "text": text.capitalize(),
                    "emoji": emoji,
                    "discovered": False
                })
        
        # Create the data object
        data = {
            "elements": elements,
            "darkMode": True
        }
        
        # Generate the JavaScript code
        js_code = f"""
    // Element Combiner Data
    const elementData = {json.dumps(data, indent=2)};

    // Store in localStorage
    localStorage.setItem('infinite-craft-data', JSON.stringify(elementData));

    // Verification code (optional)
    const stored = localStorage.getItem('infinite-craft-data');
    console.log('Stored data:', JSON.parse(stored));
    """
        return js_code

def main():
    print("\nEnhanced Element Combiner - Create new elements by combining existing ones!")
    print("Now with emoji support! 🎮")
    print("\nChoose an option:")
    print("1. Start new model")
    print("2. Load existing model")
    print("3. Gen element code")
    print("4. Auto train multi combination")
    print("5. Get model stats")
    choice = input("Enter choice (1-5): ")
    
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
    elif choice == "3":
        # Handle code generation
        print("\nSelect a model to generate code from:")
        available_models = ElementCombiner.list_available_models()
        
        if not available_models:
            print("No saved models found. Starting with default elements...")
            combiner = ElementCombiner()
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            model_choice = input("\nEnter model number or name (or press Enter for default): ")
            
            if not model_choice:
                combiner = ElementCombiner()
            else:
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(available_models):
                        model_name = available_models[idx]
                    else:
                        model_name = model_choice
                except ValueError:
                    model_name = model_choice
                
                try:
                    combiner = ElementCombiner.load_state(model_name)
                except FileNotFoundError:
                    print(f"Model not found. Using default elements...")
                    combiner = ElementCombiner()
        
        # Generate and display the code
        js_code = ElementCombiner.generate_js_code(combiner)
        print("\nGenerated JavaScript Code:")
        print(js_code)
        
        # Offer to save to file
        save_choice = input("\nWould you like to save this code to a file? (yes/no): ").lower()
        if save_choice == 'yes':
            filename = input("Enter filename (default: element_data.js): ").strip() or "element_data.js"
            with open(filename + ".js", 'w', encoding='utf-8') as f:
                f.write(js_code)
            print(f"\nCode saved to {filename}")
        
        return
    elif choice == "4":
        # Handle auto training
        print("\nSelect a model to auto train:")
        available_models = ElementCombiner.list_available_models()
        
        if not available_models:
            print("No saved models found. Starting with default elements...")
            combiner = ElementCombiner()
            model = combiner.create_model()
            combiner.train_model(model)
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            model_choice = input("\nEnter model number or name (or press Enter for default): ")
            
            if not model_choice:
                combiner = ElementCombiner()
                model = combiner.create_model()
                combiner.train_model(model)
            else:
                try:
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
                except FileNotFoundError:
                    print("Model not found. Starting with default elements...")
                    combiner = ElementCombiner()
                    model = combiner.create_model()
                    combiner.train_model(model)
        
        # Get auto training parameters
        amount = int(input("\nHow many combinations do you want to train? "))
        auto_pass = input("Auto pass combinations? (yes/no): ").lower() == 'yes'
        
        # Start auto training
        model = combiner.auto_train(model, amount, auto_pass)
        
        # Ask to save the model
        save_choice = input("\nWould you like to save the trained model? (yes/no): ").lower()
        if save_choice == 'yes':
            model_name = input("Enter model name: ")
            combiner.save_state(model, model_name)
        
        return
    elif choice == "5":
        # Handle stats display
        print("\nSelect a model to analyze:")
        available_models = ElementCombiner.list_available_models()
        
        if not available_models:
            print("No saved models found. Using default model...")
            combiner = ElementCombiner()
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            model_choice = input("\nEnter model number or name (or press Enter for default): ")
            
            if not model_choice:
                combiner = ElementCombiner()
            else:
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(available_models):
                        model_name = available_models[idx]
                    else:
                        model_name = model_choice
                except ValueError:
                    model_name = model_choice
                try:
                    combiner = ElementCombiner.load_state(model_name)
                except FileNotFoundError:
                    print(f"Model not found. Using default model...")
                    combiner = ElementCombiner()
        # Get and display stats
        stats = combiner.get_stats()
        
        print("\n📊 Model Statistics 📊")
        print("=" * 40)
        print(f"Total Elements: {stats['total_elements']}")
        print(f"├─ Base Elements: {stats['base_elements']}")
        print(f"└─ Derived Elements: {stats['derived_elements']}")
        print(f"\nTotal Combinations: {stats['total_combinations']}")
        print(f"Total Invalid Combinations: {stats['total_invalid_combinations']}")
        
        print("\nMost Versatile Elements:")
        for elem, count in stats['most_versatile_elements']:
            emoji = combiner.element_emojis.get(elem, '')
            print(f"├─ {elem} {emoji}: {count} combinations")
        
        return
    else:
        combiner = ElementCombiner()
        model = combiner.create_model()
        combiner.train_model(model)
    
    while True:
        print("\nEnter two elements to combine (or 'quit' to exit, 'save' to save model):")
        print("\nAvailable base elements:")
        for elem, emoji in combiner.base_elements.items():
            print(f"{elem} {emoji}")
        
        print("\nKnown elements:")
        for elem in sorted(combiner.all_elements):
            emoji = combiner.element_emojis.get(elem, '')
            print(f"{elem} {emoji}")
            
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
        result, emoji, confidence = combiner.predict_combination(model, elem1, elem2)
        
        if result:
            print(f"\nPredicted result: {result} {emoji}")
            
            if confidence == 1.0:
                print("(This is a known combination)")
            else:
                print("(This is a suggested new combination from the API)")
            print(f"Confidence: {confidence:.2%}")
            
            # Ask if this combination exists
            agree_result = input("\nDo you agree with this result? (yes/no/skip): ").lower()
            
            if agree_result == 'skip':
                pass
            elif agree_result == 'no':
                correct_result = input("What is the correct result? ").lower()
                correct_emoji = input("What emoji should represent this result? ")
                
                # Add new combination to training data
                combiner.add_combination(elem1, elem2, correct_result, correct_emoji)
                
                # Recreate and retrain model with updated data
                model = combiner.create_model()
                combiner.train_model(model)
                print("\nCombination added to known combinations!")
            elif agree_result == 'yes' and confidence < 1.0:
                # Add the suggested combination
                combiner.add_combination(elem1, elem2, result, emoji)
                
                # Recreate and retrain model with updated data
                model = combiner.create_model()
                combiner.train_model(model)
                print("\nSuggested combination added to known combinations!")
        else:
            print("\nFailed to generate combination. Please try again.")
        
        print("\nKnown combinations:")
        for combo, (res, emoji) in combiner.combinations:
            elements = list(combo)
            print(f"{elements[0]} {combiner.element_emojis.get(elements[0], '')} + "
                  f"{elements[1]} {combiner.element_emojis.get(elements[1], '')} = "
                  f"{res} {emoji}")



if __name__ == "__main__":
    main()