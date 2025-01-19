class StatsManager:
    def __init__(self, element_manager):
        self.element_manager = element_manager
    
    def get_stats(self):
        """Get statistics about the current model state"""
        stats = {
            'total_elements': len(self.element_manager.all_elements),
            'base_elements': len(self.element_manager.base_elements),
            'derived_elements': len(self.element_manager.all_elements) - len(self.element_manager.base_elements),
            'total_combinations': len(self.element_manager.combinations),
            'total_invalid_combinations': len(self.element_manager.invalid_combinations),
            'most_versatile_elements': self._get_most_versatile_elements(),
        }
        return stats
        
    def _get_most_versatile_elements(self):
        """Find elements that appear in the most combinations"""
        element_counts = {}
        for combo, _ in self.element_manager.combinations:
            for elem in combo:
                element_counts[elem] = element_counts.get(elem, 0) + 1
        
        # Get top 3 most used elements
        sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
        return [(elem, count) for elem, count in sorted_elements[:3]]