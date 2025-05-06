# CoRe: A Comparative Study of Transformer Models for Context Recognition in Smart Classrooms

CoRe is a comparative study of transformer models for context recognition in K-12 smart classrooms. A custom dataset is curated that comprises short commands and conversations related to day-to-day classroom operations. The dataset is utilized to train multiple transformer and neural network models to identify the best-performing model for recognizing context. To ensure data quality and support data-driven decision-making, CoRe performs topic modeling technique. The collected data is categorized into two categories: academic and non-academic. Academic data consists of instructional and inquiry-based statements, such as "Make a report on class attendance" or "Could you please explain algebra?" Each command was assigned a corresponding label. Similarly, non-academic data includes statements like "It is too noisy outside" or "Set a timer for 10 minutes", which were also labeled accordingly. The dataset comprises 770 unique sentences categorized into 14 distinct labels. The evaluation results show that a transformer model enhanced with an attention mechanism outperforms existing transformer models while maintaining low computational costs, making it a viable solution for real-world smart classroom applications.

# Getting Started

To get started with this project, follow these steps:

Clone the Repository: git clone https://github.com/Irfan995/CoRe.git

### Install Dependencies

Create and activate virtual environments

Navigate to the project directory and install the necessary dependencies using pip install -r requirements.txt.

### Run the System

  Step 1: Navigate to project directory CoRe
  
  Step 2: Run main.py (This file only runs Transformer models)
  
  Optional: Change model_type in main.py to run with different model. 
  
  Model types: bert, sbert and saf (Keep the string in this format)

  Step 3: Run deep_neural_network.py (Run models without any encoder)

  Run deep_neural_network_word2vec.py (Run models with word2vec encoder)

  Run deep_neural_network_glove.py (Run models with glove encoder)
  
# Contribution

We welcome contributions from the community. If you are interested in contributing, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

# License

Licensed under the MIT License. See the LICENSE file for details.

# Citation

If you use this code or dataset in your research, please cite:

FA Irfan, R Iqbal, D Eckstein and M Strickland, "CoRe: A Comparative Study of Transformer Models for Context Recognition in Smart Classrooms," 2025 IEEE 49th International Conference on Computers, Software, and Applications (COMPSAC), Toronto, Canada, 2025.
