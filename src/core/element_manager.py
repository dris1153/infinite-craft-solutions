import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder # type: ignore

from ..api.combination_api import CombinationAPI
from ..utils.config import EPOCHS_CONFIG, BATCH_SIZE_CONFIG
from ..models.element_model import ElementModel

class ElementManager:
    def __init__(self):
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
        
        self.initialize_elements()
        
    def initialize_elements(self):
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
        
    def get_state(self):
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
        return state

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
        
    def add_invalid_combination(self, elem1, elem2):
        """Add a combination that's known to be invalid"""
        combo_key = self.get_combination_key(elem1, elem2)
        self.invalid_combinations.add(combo_key)
        
    def is_invalid_combination(self, elem1, elem2):
        """Check if a combination is known to be invalid"""
        combo_key = self.get_combination_key(elem1, elem2)
        return combo_key in self.invalid_combinations

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
        new_element, emoji = CombinationAPI.get_combination(element1, element2, self.add_invalid_combination)
        if new_element and emoji:
            return new_element, emoji, 0.5
        elif new_element is None:  # API indicated invalid combination
            return None, None, 0.5
        
        # Fallback to model prediction if API fails
        return None, None, 0.0

    def prepare_data(self, batch_size=BATCH_SIZE_CONFIG):
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
            
            if invalid_combinations_index % 1000 == 0:
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
    
    def get_train_elements(self, uncombined, index):
        # Prioritize uncombined pairs
        chosen_pair = uncombined[index]
        return chosen_pair[0], chosen_pair[1]
      
    def train_model(self, model, epochs=EPOCHS_CONFIG, batch_size=BATCH_SIZE_CONFIG):
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
            model = ElementModel.create_model(
                input_dim=len(self.all_elements) * 2,
                output_dim=len(self.all_elements)
            )
            self.train_model(model)  
        except KeyboardInterrupt:
            print("\n\nAuto training interrupted by user")
        
        return model