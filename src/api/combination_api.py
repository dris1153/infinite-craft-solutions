import requests

class CombinationAPI:
    @staticmethod
    def get_combination(elem1, elem2, add_invalid_combination):
        """Get combination result from the API"""
        try:
            response = requests.get(f'https://infiniteback.org/pair?first={elem1.capitalize()}&second={elem2.capitalize()}')
            if response.status_code == 200:
                data = response.json()
                print("+++",data,"+++")
                if data is not None:
                    return data['result'].lower(), data['emoji']
                else:
                    add_invalid_combination(elem1, elem2)
                    return None, None
            return None, None
        except Exception as e:
            print(f"API Error: {e}")
            return None, None