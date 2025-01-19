from .ui.cli import CLI
from .core.element_manager import ElementManager
from .models.element_model import ElementModel
from .storage.model_storage import ModelStorage
from .utils.config import DATA_PATH

def main():
    element_manager = ElementManager()
    model_storage = ModelStorage(DATA_PATH)
    cli = CLI(element_manager, model_storage)
    cli.start()

if __name__ == "__main__":
    main()