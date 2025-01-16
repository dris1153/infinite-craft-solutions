import nltk # type: ignore

from .classes.ElementCombiner import ElementCombiner

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


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