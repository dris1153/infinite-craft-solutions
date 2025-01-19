import json

class CodeGenerator:
    def __init__(self, element_manager):
        self.element_manager = element_manager
    
    def generate_js_code(self):
        """Generate JavaScript code for localStorage"""
        elements = []
        
        # Add base elements
        for text, emoji in self.element_manager.base_elements.items():
            elements.append({
                "text": text.capitalize(),
                "emoji": emoji,
                "discovered": False
            })
        
        # Add combined elements
        for element in self.element_manager.all_elements:
            if element not in self.element_manager.base_elements:  # Avoid duplicates
                emoji = self.element_manager.element_emojis.get(element, "❓")
                elements.append({
                    "text": element.capitalize(),
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
        
    def save_to_file(self, js_code, filename):
        """Save generated code to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(js_code)