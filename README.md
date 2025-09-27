# Law Advisor with LLaMA 3.2

Welcome to the **Law Advisor** repository! This project leverages the LLaMA 3.2 model fine-tuned on legal datasets to provide a robust legal advisory system. The system is designed to assist with legal queries, offering insights and information based on Indian laws and regulations.

## Features

- **Fine-tuned LLaMA 3.2 Model**: Optimized for legal advisory tasks.
- **Support for Indian Legal Datasets**: Includes IndicLegalQA, Indian Law, IPC Sections, and the Constitution of India.
- **Modular Design**: Scripts for data preparation, training, merging, and inference.
- **Efficient Inference**: Supports low-rank adaptation (LoRA) for resource efficiency.

## Repository Structure

- **`prep_data_phase1.py` & `prep_data_phase2.py`**: Script for preparing the data for training.
- **`train_phase1.py` & `train_phase2.py`**: Script for training the model.
- **`merge_phase1.py` & `merge_phase2.py`**: Script for merging models in the first and second phase.
- **`interface.py`**: Interface for interacting with the fine-tuned model.
- **`data/`**: Contains datasets used for fine-tuning.
- **`outputs/` & `outputs_phase2/`**: Stores checkpoints and final models.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/py-raj/Law-Advisor.git
   cd Law-Advisor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the project:
   - Update cfg with the required settings in `train_phase1.py` & `train_phase2.py`.
   - Place your Hugging Face token in HUGGINGFACE_TOKEN.

4. Prepare the data:
   ```bash
   python prep_data_phase1.py
   ```

5. Prepare the data:
   ```bash
   python prep_data_phase2.py
   ```

6. Train the model:
   ```bash
   python train_phase1.py
   ```

7. Merge models:
   ```bash
   python merge_phase1.py
   ```

8. Train the model:
   ```bash
   python train_phase2.py
   ```

9. Merge models:
   ```bash
   python merge_phase2.py
   ```

9. Run the interface:
   ```bash
   python interface.py
   ```

## Dataset Credits

This project utilized the following datasets for fine-tuning the model:

- **IndicLegalQA**: [Dataset on Mendeley](https://data.mendeley.com/datasets/gf8n8cnmvc/2)
- **Indian Law**: [Dataset on Kaggle](https://www.kaggle.com/datasets/anishparkhe0401/indian-laws)
- **Indian Penal Code (IPC) Sections Information**: [Dataset on Kaggle](https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information)
- **Constitution Of India**: [Dataset on Kaggle](https://www.kaggle.com/datasets/rushikeshdarge/constitution-of-india)

## Outputs

- **`outputs/`**: Contains checkpoints and final models from the first training phase.

- **`outputs_phase2/`**: Contains checkpoints and final models from the second training phase.

## Data Folder

The `data/` folder contains the datasets used for fine-tuning the model and the prepared data for `train_phase1.py` & `train_phase2.py`. It includes:

- **`constitution.csv`**: Data related to the Constitution of India.
- **`indian_law.json`**: JSON file containing Indian legal data.
- **`ipc.csv`**: Information about Indian Penal Code (IPC) sections.
- **`indiclegalqa.json`**: Dataset for Indic Legal QA tasks.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments

- Hugging Face for providing the LLaMA model.
- Creators of the datasets for their invaluable contributions.

---

Feel free to reach out for any questions or issues!
