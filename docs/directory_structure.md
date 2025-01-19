element-combiner/
├── api/
│ ├── **init**.py
│ └── combination_api.py # Handles API requests to infiniteback.org
│
├── models/
│ ├── **init**.py
│ ├── element_model.py # Neural network model definition and training
│ └── data_preparation.py # Data preparation for model training
│
├── storage/
│ ├── **init**.py
│ ├── model_storage.py # Model saving/loading functionality
│ └── data_storage.py # JSON/pickle data storage handling
│
├── core/
│ ├── **init**.py
│ ├── element_manager.py # Main element combination logic
│ ├── element_validator.py # Validation of combinations
│ └── stats_manager.py # Statistics generation
│
├── ui/
│ ├── **init**.py
│ ├── cli.py # Command line interface
│ └── code_generator.py # JavaScript code generation
│
├── utils/
│ ├── **init**.py
│ └── config.py # Configuration constants
│
├── data/ # Storage directory for saved models
│ └── .gitkeep
│
└── main.py # Entry point
