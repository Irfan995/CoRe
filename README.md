# CoRe: A Comparative Study of Transformer Models for Context Recognition in Smart Classrooms

![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A comparative study of transformer models for context recognition in K-12 smart classrooms. CoRe curates a custom dataset of short commands and conversations related to day-to-day classroom operations, performs topic modeling to separate academic from non-academic utterances, and benchmarks several transformer-based and neural-network models to identify the best performer under real-world constraints.

---

## Table of Contents

- [Dataset](#dataset)  
- [Topic Modeling](#topic-modeling)  
- [Evaluation](#evaluation)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  
- [Citation](#citation)  

---

## Dataset

- **Size:** 770 unique smart short commands  
- **Labels:** 14 distinct context categories  
- **Categories:**  
  - **Academic:** instructional or inquiry statements  
    - e.g. “Make a report on class attendance”  
    - e.g. “Could you please explain algebra?”  
  - **Non-Academic:** other operational or ambient commands  
    - e.g. “It is too noisy outside”  
    - e.g. “Set a timer for 10 minutes”  

Each sentence in the dataset has been manually labeled to support supervised training.

---

## Topic Modeling

To ensure data quality and provide insights into usage patterns, CoRe applies a topic-modeling technique (e.g., LDA) to automatically separate academic from non-academic data prior to model training.

---

## Evaluation

Benchmarking shows that a transformer model enhanced with an attention mechanism:

- **Outperforms** standard transformer architectures  
- **Maintains** low computational cost  
- **Is viable** for real-time smart-classroom deployments  

---

## Getting Started

### Prerequisites

- Python 3.8  
- `virtualenv` or `venv` (optional, but recommended)  

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Irfan995/CoRe.git
   cd CoRe
   ```

2. **Create & activate** a virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Transformer Models

Run the main script to train and evaluate transformer variants:

```bash
python main.py
```

> By default, `main.py` runs all supported transformer types.  
> To specify a model, edit the `model_type` variable in `main.py` to one of:  
> `bert`, `sbert`, or `saf`.

### 2. Deep Neural Network Models

- **Without encoder:**  
  ```bash
  python deep_neural_network.py
  ```
- **With Word2Vec encoder:**  
  ```bash
  python deep_neural_network_word2vec.py
  ```
- **With GloVe encoder:**  
  ```bash
  python deep_neural_network_glove.py
  ```

---

## Contributing

We welcome community contributions!  

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add YourFeature"`)  
4. Push to your branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

For substantial changes, please open an issue to discuss your proposal before coding.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or dataset in your research, please cite:

> F. A. Irfan, R. Iqbal, D. Eckstein, and M. Strickland, “CoRe: A Comparative Study of Transformer Models for Context Recognition in Smart Classrooms,” in _2025 IEEE 49th International Conference on Computers, Software, and Applications (COMPSAC)_, Toronto, Canada, 2025.
