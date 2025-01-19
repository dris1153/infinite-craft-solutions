from ..models.element_model import ElementModel
from ..core.element_manager import ElementManager
from ..storage.model_storage import ModelStorage
from ..core.stats_manager import StatsManager

from .code_generator import CodeGenerator

class CLI:
    def __init__(self, element_manager: ElementManager, model_storage: ModelStorage):
        self.element_manager = element_manager
        self.model_storage = model_storage
        self.stats_manager = StatsManager(element_manager)
        self.code_generator = CodeGenerator(element_manager)
        self.current_model = None
    
    def start(self):
        print("\nEnhanced Element Combiner - Create new elements by combining existing ones!")
        print("Now with emoji support! 🎮")
        print("\nChoose an option:")
        print("1. Start new model")
        print("2. Load existing model")
        print("3. Gen element code")
        print("4. Auto train multi combination")
        print("5. Get model stats")
        choice = input("Enter choice (1-5): ")
        
        if choice == "1":
            self._handle_new_model()
        elif choice == "2":
            self._handle_choose_model()
            self._enter_combination_loop()
        elif choice == "3":
            self._handle_choose_model()
            self._handle_code_generation()
        elif choice == "4":
            self._handle_choose_model()
            self._handle_auto_training()
        elif choice == "5":
            self._handle_choose_model()
            self._handle_stats()
        else:
            print("Invalid choice. Starting new model...")
            self._handle_new_model()
            
    def _handle_choose_model(self):
        """Handle displaying model statistics"""
        print("\nSelect a model to analyze:")
        available_models = self.model_storage.list_available_models()
        
        if not available_models:
            print("No saved models found. Using default model...")
            self._handle_new_model()
            return
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            model_choice = input("\nEnter model number or name (or press Enter for default): ")
            
            if model_choice:
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(available_models):
                        model_name = available_models[idx]
                    else:
                        model_name = model_choice
                    try:
                        self.element_manager = self.model_storage.load_state(model_name)
                        self.current_model = self.model_storage.load_model(model_name)
                        self.code_generator = CodeGenerator(self.element_manager)
                        self.stats_manager = StatsManager(self.element_manager)
                    except FileNotFoundError:
                        print(f"Model not found. Using default model...")
                except ValueError:
                    print("Invalid choice. Using default model...")
    
    def _handle_new_model(self):
        """Handle creation of new model"""
        self.current_model = ElementModel.create_model(
            input_dim=len(self.element_manager.all_elements) * 2,
            output_dim=len(self.element_manager.all_elements)
        )
        self.element_manager.train_model(self.current_model)
        self._enter_combination_loop()
            
    def _handle_code_generation(self):
        """Handle JavaScript code generation"""
        js_code = self.code_generator.generate_js_code()
        print("\nGenerated JavaScript Code:")
        print(js_code)
        
        save_choice = input("\nWould you like to save this code to a file? (yes/no): ").lower()
        if save_choice == 'yes':
            filename = input("Enter filename (default: element_data.js): ").strip() or "element_data.js"
            self.code_generator.save_to_file(js_code, filename)
            print(f"\nCode saved to {filename}")
            
    def _handle_auto_training(self):
        """Handle automated training process"""            
        amount = int(input("\nHow many combinations do you want to train? "))
        auto_pass = input("Auto pass combinations? (yes/no): ").lower() == 'yes'
        
        self.current_model = self.element_manager.auto_train(self.current_model, amount, auto_pass)
        
        save_choice = input("\nWould you like to save the trained model? (yes/no): ").lower()
        if save_choice == 'yes':
            model_name = input("Enter model name: ")
            self.model_storage.save_state(self.current_model, model_name, self.element_manager.get_state(), self.element_manager.element_encoder)
            
    def _handle_stats(self):              
        stats = self.stats_manager.get_stats()
        self._display_stats(stats)

    def _display_stats(self, stats):
        """Display model statistics"""
        print("\n📊 Model Statistics 📊")
        print("=" * 40)
        print(f"Total Elements: {stats['total_elements']}")
        print(f"├─ Base Elements: {stats['base_elements']}")
        print(f"└─ Derived Elements: {stats['derived_elements']}")
        print(f"\nTotal Combinations: {stats['total_combinations']}")
        print(f"Total Invalid Combinations: {stats['total_invalid_combinations']}")
        
        print("\nMost Versatile Elements:")
        for elem, count in stats['most_versatile_elements']:
            emoji = self.element_manager.element_emojis.get(elem, '')
            print(f"├─ {elem} {emoji}: {count} combinations")

    def _enter_combination_loop(self):
        """Main combination input loop"""
        while True:
            print("\nEnter two elements to combine (or 'quit' to exit, 'save' to save model):")
            self._display_available_elements()
            
            elem1 = input("First element: ").lower()
            
            if elem1 == 'quit':
                break
            elif elem1 == 'save':
                model_name = input("Enter model name: ")
                self.model_storage.save_state(self.current_model, model_name, self.element_manager.get_state(), self.element_manager.element_encoder)
                continue
            
            elem2 = input("Second element: ").lower()
            if elem2 == 'quit':
                break
            
            self._process_combination(elem1, elem2)

    def _display_available_elements(self):
        """Display available elements to the user"""
        print("\nAvailable base elements:")
        for elem, emoji in self.element_manager.base_elements.items():
            print(f"{elem} {emoji}")
        
        print("\nKnown elements:")
        for elem in sorted(self.element_manager.all_elements):
            emoji = self.element_manager.element_emojis.get(elem, '')
            print(f"{elem} {emoji}")

    def _process_combination(self, elem1: str, elem2: str):
        """Process a combination attempt"""
        result, emoji, confidence = self.element_manager.predict_combination(self.current_model, elem1, elem2)
        
        if result:
            print(f"\nPredicted result: {result} {emoji}")
            
            if confidence == 1.0:
                print("(This is a known combination)")
            else:
                print("(This is a suggested new combination from the API)")
            print(f"Confidence: {confidence:.2%}")
            
            agree_result = input("\nDo you agree with this result? (yes/no/skip): ").lower()
            
            if agree_result == 'skip':
                return
            elif agree_result == 'no':
                correct_result = input("What is the correct result? ").lower()
                correct_emoji = input("What emoji should represent this result? ")
                
                self.element_manager.add_combination(elem1, elem2, correct_result, correct_emoji)
                self.current_model = ElementModel.create_model(
                    input_dim=len(self.element_manager.all_elements) * 2,
                    output_dim=len(self.element_manager.all_elements)
                )
                self.element_manager.train_model(self.current_model)
                print("\nCombination added to known combinations!")
            elif agree_result == 'yes' and confidence < 1.0:
                self.element_manager.add_combination(elem1, elem2, result, emoji)
                self.current_model = ElementModel.create_model(
                    input_dim=len(self.element_manager.all_elements) * 2,
                    output_dim=len(self.element_manager.all_elements)
                )
                self.element_manager.train_model(self.current_model)
                print("\nSuggested combination added to known combinations!")
        else:
            print("\nFailed to generate combination. Please try again.")
            
        self._display_known_combinations()
    
    def _display_known_combinations(self):
        """Display all known combinations"""
        print("\nKnown combinations:")
        for combo, (res, emoji) in self.element_manager.combinations:
            elements = list(combo)
            print(f"{elements[0]} {self.element_manager.element_emojis.get(elements[0], '')} + "
                  f"{elements[1]} {self.element_manager.element_emojis.get(elements[1], '')} = "
                  f"{res} {emoji}")