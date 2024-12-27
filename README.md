# IndoML Datathon 2024 Phase 2 - Final Submission

## Competition Details
This repository contains the final submission for the IndoML Datathon 2024 Phase 2. The competition was hosted on Codalab, and more details can be found [here](https://codalab.lisn.upsaclay.fr/competitions/20229#learn_the_details).

## Achievement
We are proud that our solution won the second prize in the competition. We presented our solution on the third day of the IndoML 2024 event, held on 23rd December 2024.

## Presentation
The presentation of our solution can be accessed [here](https://www.canva.com/design/DAGZomUQk10/GNKWN0CTtTn848EyRxHEvQ/edit?utm_content=DAGZomUQk10&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## Repository Structure
```
├── data/
│   └── Contains the augmented dataset used for training and evaluation.
├── tuning_t5.py
│   └── Script for tuning the T5 model.
├── entrypoint.sh
│   └── Entrypoint script for Docker.
├── Dockerfile
│   └── Dockerfile to set up the environment.
├── requirements.txt
│   └── List of dependencies required for the project.
├── logs/
│   └── Directory containing training logs.
├── Other Attempts/
│   └── Contains notebooks from other attempts.
├── Efficiency.ipynb
│   └── Notebook for efficiency analysis.
└── Data_Analysis.ipynb
  └── Notebook for data analysis.
```

## How to Run
1. **Clone the repository:**
  ```sh
  git clone https://github.com/Matrixmang0/IndoML-Datathon-2024-P2.git
  cd IndoML-Datathon-2024-P2
  ```

2. **Build the Docker image:**
  ```sh
  docker build -t indoml_datathon_2024 .
  ```

3. **Run the Docker container:**
  ```sh
  docker run -it indoml_datathon_2024
  ```

## Contact
For any queries, please contact us at [santhoshgs013@gmail.com].
