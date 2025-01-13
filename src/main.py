import numpy as np
import json
import pickle
import os
import nltk # type: ignore
from nltk.corpus import words, wordnet # type: ignore
from nltk.tag import pos_tag # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import random

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datas"))

class ElementCombiner:
    def __init__(self):
        # Define base elements
        self.base_elements = ['fire', 'water', 'wind', 'earth']
        
        # Define known combinations and results
        self.combinations = {
            (frozenset(['water', 'fire']), 'steam'),
            (frozenset(['water', 'earth']), 'mud'),
            (frozenset(['fire', 'earth']), 'metal'),
            (frozenset(['wind', 'water']), 'mist')
        }
        
        # Load English words
        self.english_words = set(word.lower() for word in words.words())
        self._initialize_elements()
    
    def _initialize_elements(self):
        # Create a set of all unique elements
        self.all_elements = set(self.base_elements)
        for combo, result in self.combinations:
            self.all_elements.add(result)
            self.all_elements.update(combo)
        
        self.all_elements = list(sorted(self.all_elements))  # Sort for consistency
        
        # Initialize label encoders
        self.element_encoder = LabelEncoder()
        self.element_encoder.fit(self.all_elements)

    def save_state(self, model, model_name):
        """Save model, combinations, and encoder state"""
        base_path = os.path.join(data_path, model_name)
        os.makedirs(base_path, exist_ok=True)
        
        model.save(os.path.join(base_path, "model.keras"))
        
        # Convert frozensets to lists for JSON serialization
        serializable_combinations = [
            ([list(combo), result]) for combo, result in self.combinations
        ]
        
        state = {
            'base_elements': self.base_elements,
            'combinations': serializable_combinations,
            'all_elements': self.all_elements
        }
        
        with open(os.path.join(base_path, "data.json"), 'w') as f:
            json.dump(state, f, indent=4)
            
        with open(os.path.join(base_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(self.element_encoder, f)
            
        print(f"Model saved in folder: {base_path}")
            
    @classmethod
    def load_state(cls, model_name):
        """Load a saved state and return a new ElementCombiner instance"""
        base_path = os.path.join(data_path, model_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model folder not found: {base_path}")
            
        instance = cls.__new__(cls)
        
        # Load combinations and elements
        with open(os.path.join(base_path, "data.json"), 'r') as f:
            state = json.load(f)
            
        instance.base_elements = state['base_elements']
        # Convert lists back to frozensets
        instance.combinations = {
            (frozenset(combo), result) for [combo, result] in state['combinations']
        }
        instance.all_elements = state['all_elements']
        
        # Load English words
        instance.english_words = set(word.lower() for word in words.words())
        
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

    def get_combination_result(self, elem1, elem2):
        """Get the known result for a combination of elements"""
        combo = frozenset([elem1, elem2])
        for known_combo, result in self.combinations:
            if combo == known_combo:
                return result
        return None
    
    def add_combination(self, elem1, elem2, result):
        """Add a new combination to the training data, replacing any existing combination"""
        combo = frozenset([elem1, elem2])
        # Remove any existing combination with the same elements
        self.combinations = {c for c in self.combinations if c[0] != combo}
        self.combinations.add((combo, result))
        
        # Update all_elements if new elements were introduced
        new_elements = {elem1, elem2, result}
        if not new_elements.issubset(self.all_elements):
            self.all_elements.extend(sorted(list(new_elements - set(self.all_elements))))
            self.element_encoder.fit(self.all_elements)
    
    def get_related_words(self, word1, word2):
        """Get related English nouns based on WordNet relationships"""
        related_words = set()
        
        # Get synsets for both words (nouns only)
        synsets1 = wordnet.synsets(word1, pos=wordnet.NOUN)
        synsets2 = wordnet.synsets(word2, pos=wordnet.NOUN)
        
        for syn1 in synsets1:
            # Get hypernyms (more general terms)
            for hypernym in syn1.hypernyms():
                related_words.update(lemma.name().lower() for lemma in hypernym.lemmas())
            
            # Get hyponyms (more specific terms)
            for hyponym in syn1.hyponyms():
                related_words.update(lemma.name().lower() for lemma in hyponym.lemmas())
                
        for syn2 in synsets2:
            for hypernym in syn2.hypernyms():
                related_words.update(lemma.name().lower() for lemma in hypernym.lemmas())
            for hyponym in syn2.hyponyms():
                related_words.update(lemma.name().lower() for lemma in hyponym.lemmas())
        
        # Helper function to check if a word is a noun
        def is_noun(word):
            tagged = pos_tag([word])
            return tagged[0][1].startswith('NN')
        
        # Filter out multi-word expressions, non-alphabet words, and non-nouns
        related_words = {word for word in related_words 
                        if word.isalpha() and '_' not in word and is_noun(word)}
        
        return related_words

    def generate_new_element(self, elem1, elem2):
        """Generate a new element name based on the combination of two elements (nouns only)"""
        # First try to get semantically related nouns
        related_words = self.get_related_words(elem1, elem2)
        
        # Filter words that are not already elements
        existing_elements = {result for _, result in self.combinations}
        existing_elements.update(self.base_elements)
        
        candidate_words = related_words - existing_elements
        
        if candidate_words:
            # Prefer shorter words (length 4-8 characters)
            preferred_words = [word for word in candidate_words 
                             if 4 <= len(word) <= 8]
            if preferred_words:
                return random.choice(preferred_words)
            return random.choice(list(candidate_words))
        
        # Fallback: find noun words that share some letters with both elements
        common_letters = set(elem1).intersection(set(elem2))
        if common_letters:
            # Get all English words that are nouns
            english_nouns = {word for word in self.english_words 
                           if pos_tag([word])[0][1].startswith('NN')}
            
            candidates = [word for word in english_nouns 
                        if any(letter in word for letter in common_letters)
                        and word not in existing_elements
                        and 4 <= len(word) <= 8]
            if candidates:
                return random.choice(candidates)
        
        # Final fallback: return a random English noun
        while True:
            word = random.choice(list(self.english_words))
            if (word not in existing_elements and 
                4 <= len(word) <= 8 and 
                pos_tag([word])[0][1].startswith('NN')):
                return word
    
    def prepare_data(self):
        """Prepare training data for the model"""
        X = []  # Input combinations
        y = []  # Results
        
        # Convert combinations to training data
        for combo, result in self.combinations:
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

    def predict_combination(self, model, element1, element2):
        _ = model  # Access the model parameter to avoid unused parameter warning
        """Predict the result of combining two elements"""
        # First check if this is a known combination
        known_result = self.get_combination_result(element1, element2)
        if known_result is not None:
            return known_result, 1.0
        
        # For new combinations, generate a new element name
        new_element = self.generate_new_element(element1, element2)
        return new_element, 0.5  # Use 0.5 confidence for new generations

def main():
    print("\nEnhanced Element Combiner - Create new elements by combining existing ones!")
    print("This version creates meaningful noun-based elements!")
    print("\nChoose an option:")
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
        combiner = ElementCombiner()
        model = combiner.create_model()
        combiner.train_model(model)
    
    while True:
        print("\nEnter two elements to combine (or 'quit' to exit, 'save' to save model):")
        print("Available base elements:", ", ".join(combiner.base_elements))
        print("Known elements:", ", ".join(sorted(combiner.all_elements)))
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
        
        if confidence == 1.0:
            print("(This is a known combination)")
        else:
            print("(This is a suggested new combination using a related English noun)")
        print(f"Confidence: {confidence:.2%}")
        
        # Ask if this combination exists
        agree_result = input("\nDo you agree with this result? (yes/no/skip): ").lower()
        
        if agree_result == 'skip':
            pass
        elif agree_result == 'no':
            correct_result = input("What is the correct result? ").lower()
            
            # Add new combination to training data
            combiner.add_combination(elem1, elem2, correct_result)
            
            # Recreate and retrain model with updated data
            model = combiner.create_model()
            combiner.train_model(model)
            print("\nCombination added to known combinations!")
        else:
            # Only for new combinations
            if confidence < 1.0:
                keep_suggestion = input("Would you like to keep the suggested combination? (yes/no): ").lower()
                if keep_suggestion == 'yes':
                    # Add the suggested combination
                    combiner.add_combination(elem1, elem2, result)
                    
                    # Recreate and retrain model with updated data
                    model = combiner.create_model()
                    combiner.train_model(model)
                    print("\nSuggested combination added to known combinations!")
        
        print("\nKnown combinations:")
        for combo, res in combiner.combinations:
            elements = list(combo)
            print(f"{elements[0]} + {elements[1]} = {res}")

if __name__ == "__main__":
    main()