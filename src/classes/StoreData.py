import os

class StoreData:
    def __init__(self, path):
        self.path = path
        self.result = {}
        if os.path.exists(self.path):
            with open(self.path, 'r') as file:
                for line in file:
                    # Split the line into key and value
                    key, value = map(str.strip, line.split(':'))
                    
                    # Remove single quotes from the key and value
                    key = key.strip("'")
                    value = value.strip("'")
                    
                    # Add the key-value pair to the dictionary
                    self.result[key] = value
    def getData (self, key, notValue = ''):
        if key in self.result:
            return self.result[key]
        else:
            return notValue
    def storeData(self, key, value):
        self.result[key] = value
    def saveFile(self):
        f = open(self.path, "w")
        output_array = [f"{key}:{value}" for key, value in self.result.items()]
        f.write("\n".join(output_array))
        f.close()
    def saveData(self, key, value):
        self.storeData(key,value)
        self.saveFile()