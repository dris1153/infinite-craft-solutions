import numpy as np
import json
import pickle
import os
import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import threading
import random


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
        
        # Define known combinations and results with emojis
        self.combinations = {
            (frozenset(['water', 'fire']), ('steam', '💨')),
            (frozenset(['water', 'earth']), ('plant', '🌱')),
            (frozenset(['fire', 'earth']), ('lava', '🌋')),
            (frozenset(['wind', 'water']), ('wave', '🌊')),
            (frozenset(['wind', 'fire']), ('smoke', '💨')),
            (frozenset(['wind', 'earth']), ('dust', '🌫️'))
        }
        
        self._initialize_elements()
    
    def _initialize_elements(self):
        # Create a set of all unique elements
        self.all_elements = set(self.base_elements.keys())
        self.element_emojis = dict(self.base_elements)
        
        for combo, result in self.combinations:
            result_name, result_emoji = result
            self.all_elements.add(result_name)
            self.all_elements.update(combo)
            self.element_emojis[result_name] = result_emoji
        
        self.all_elements = list(sorted(self.all_elements))
        
        # Initialize label encoders
        self.element_encoder = LabelEncoder()
        self.element_encoder.fit(self.all_elements)

    def get_api_combination(self, elem1, elem2):
        """Get combination result from the API"""
        try:
            response = requests.get(f'https://infiniteback.org/pair?first={elem1.capitalize()}&second={elem2.capitalize()}')
            if response.status_code == 200:
                data = response.json()
                return data['result'].lower(), data['emoji']
            return None, None
        except Exception as e:
            print(f"API Error: {e}")
            return None, None

    def predict_combination(self, model, element1, element2):
        """Predict the result of combining two elements"""
        # First check if this is a known combination
        known_result = self.get_combination_result(element1, element2)
        if known_result is not None:
            return known_result[0], known_result[1], 1.0
        
        # For new combinations, call the API
        new_element, emoji = self.get_api_combination(element1, element2)
        if new_element and emoji:
            return new_element, emoji, 0.5
        
        # Fallback to model prediction if API fails
        return None, None, 0.0

    def get_combination_result(self, elem1, elem2):
        """Get the known result for a combination of elements"""
        combo = frozenset([elem1, elem2])
        for known_combo, result in self.combinations:
            if combo == known_combo:
                return result
        return None
    
    def add_combination(self, elem1, elem2, result, emoji):
        """Add a new combination to the training data"""
        combo = frozenset([elem1, elem2])
        # Remove any existing combination with the same elements
        self.combinations = {c for c in self.combinations if c[0] != combo}
        self.combinations.add((combo, (result, emoji)))
        
        # Update all_elements and emojis if new elements were introduced
        if result not in self.all_elements:
            self.all_elements.append(result)
            self.element_emojis[result] = emoji
            self.element_encoder.fit(self.all_elements)

    def prepare_data(self):
        """Prepare training data for the model"""
        X = []  # Input combinations
        y = []  # Results
        
        # Convert combinations to training data
        for combo, (result, _) in self.combinations:
            # Create both orderings of elements for training
            elements = list(combo)
            pairs = [(elements[0], elements[1]), (elements[1], elements[0])]
            
            for elem1, elem2 in pairs:
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
        """Create and return a new neural network model"""
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
        """Train the model on the current combinations"""
        X, y = self.prepare_data()
        model.fit(X, y, epochs=epochs, verbose=0)

    def get_random_elements(self):
        """Get two random elements, prioritizing uncombined pairs"""
        all_elements_list = list(self.all_elements)
        
        # Create a set of all possible combinations
        all_possible = set()
        for i in range(len(all_elements_list)):
            for j in range(len(all_elements_list)):
                if (all_elements_list[i] != "" and all_elements_list[j] != "" and all_elements_list[i] != all_elements_list[j]):
                    all_possible.add(frozenset([all_elements_list[i], all_elements_list[j]]))
        
        # Get set of existing combinations
        existing_combos = {combo for combo, _ in self.combinations}
        
        # Find combinations that don't exist yet
        uncombined = list(all_possible - existing_combos)
        
        if uncombined:
            # Prioritize uncombined pairs
            t = random.choice(uncombined)
            chosen_pair = list(t)
            return chosen_pair[0], chosen_pair[1]
        else:
            # If all pairs are combined, choose random elements
            elem1 = random.choice(all_elements_list)
            elem2 = random.choice(all_elements_list)
            return elem1, elem2

    def auto_train(self, model, amount, auto_pass=False):
        """Auto train the model with random combinations"""
        trained_count = 0
        
        print(f"\nStarting auto training for {amount} combinations...")
        print("Press Ctrl+C to stop at any time\n")
        
        try:
            while trained_count < amount:
                elem1, elem2 = self.get_random_elements()
                result, emoji, confidence = self.predict_combination(model, elem1, elem2)
                
                print(f"\nTrying combination {trained_count + 1}/{amount}")
                print(f"Elements: {elem1} {self.element_emojis.get(elem1, '')} + "
                    f"{elem2} {self.element_emojis.get(elem2, '')}")
                
                if result:
                    print(f"Predicted result: {result} {emoji}")
                    print(f"Confidence: {confidence:.2%}")
                    
                    if auto_pass:
                        if confidence < 1.0:
                            self.add_combination(elem1, elem2, result, emoji)
                            model = self.create_model()
                            self.train_model(model)
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
                            model = self.create_model()
                            self.train_model(model)
                            print("\nCombination added to known combinations!")
                            trained_count += 1
                        elif agree_result == 'yes' and confidence < 1.0:
                            self.add_combination(elem1, elem2, result, emoji)
                            model = self.create_model()
                            self.train_model(model)
                            print("\nSuggested combination added to known combinations!")
                            trained_count += 1
                else:
                    print("\nFailed to generate combination. Trying another...")
                    
                print(f"\nProgress: {trained_count}/{amount} combinations trained")
                
        except KeyboardInterrupt:
            print("\n\nAuto training interrupted by user")
        
        return model
    
    def save_state(self, model, model_name):
        """Save model, combinations, and encoder state"""
        base_path = os.path.join(data_path, model_name)
        os.makedirs(base_path, exist_ok=True)
        
        model.save(os.path.join(base_path, "model.keras"))
        
        # Convert frozensets to lists for JSON serialization
        serializable_combinations = [
            ([list(combo), list(result)]) for combo, result in self.combinations
        ]
        
        state = {
            'base_elements': self.base_elements,
            'combinations': serializable_combinations,
            'all_elements': self.all_elements,
            'element_emojis': self.element_emojis
        }
        
        with open(os.path.join(base_path, "data.json"), 'w') as f:
            json.dump(state, f, indent=4)
            
        with open(os.path.join(base_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(self.element_encoder, f)
            
        print(f"Model saved in folder: {base_path}")
            
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
            (frozenset(combo), tuple(result)) for [combo, result] in state['combinations']
        }
        instance.all_elements = state['all_elements']
        
        # Handle backward compatibility for element_emojis
        if 'element_emojis' in state:
            instance.element_emojis = state['element_emojis']
        else:
            # Reconstruct element_emojis from base_elements and combinations
            instance.element_emojis = dict(instance.base_elements)
            for combo, (result, emoji) in instance.combinations:
                if isinstance(result, str):  # Handle old format where result might be just a string
                    instance.element_emojis[result] = '❓'  # Default emoji for old entries
                else:
                    instance.element_emojis[result[0]] = result[1]
        
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
    choice = input("Enter choice (1-4): ")
    
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