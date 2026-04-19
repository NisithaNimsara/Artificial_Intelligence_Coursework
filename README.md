# CM2602 Artificial Intelligence Coursework

This repository contains my individual coursework submission for **CM2602 – Artificial Intelligence**. The project brings together four major areas of AI practice: **Constraint Satisfaction Problems (CSP)**, **Ontology Engineering**, **Search Algorithms** and **Intelligent Decision Systems**.

The work is organised question by question, with the final coursework report included alongside the implementation files.

## Repository Contents

```text
.
├── Question1.xlsx
├── Question2Automotive.rdf
├── Question3.py
├── Question4.py
├── 2506755_20240281_Nisitha Nimsara.pdf
└── README.md
```

## Project Overview

### Question 1 – Constraint Satisfaction Problem using Excel Solver
This section models a weekly military patrol scheduling problem as a **Constraint Satisfaction Problem (CSP)**.

It includes:
- decision variables for assigning soldiers to patrols
- operational constraints such as minimum and maximum workload
- capability-based assignment rules for day, night, and boat patrols
- an objective function to maximise overall mission readiness
- implementation using **Microsoft Excel Solver**

### Question 2 – Automotive Ontology Engineering
This section develops an **automotive domain ontology** in **RDF/OWL** form.

It covers five knowledge branches:
- Vehicle Components
- Manufacturing Processes
- Quality Assurance and Testing
- Compliance and Regulations
- Maintenance and Security Policies

It also includes:
- a concept graph / taxonomy
- ontology classes, properties, and individuals
- five SPARQL queries written to answer selected competency questions

### Question 3 – Search Algorithms for Maze Pathfinding
This section implements and compares two search algorithms for solving a **6×6 maze pathfinding problem**:
- **Iterative Deepening Depth-First Search (IDDFS)**
- **Best First Search**

The solution supports:
- random generation of start, goal, and barrier nodes
- horizontal, vertical, and diagonal movement
- path cost calculation
- repeated experiments across three random mazes
- comparison using multiple heuristics:
  - Chebyshev Distance
  - Manhattan Distance
  - Euclidean Distance
  - Diagonal Distance

### Question 4 – AMT Anomaly Detection and Correction
This section focuses on **anomaly detection and correction in Automated Manual Transmission (AMT) systems** using three different AI approaches:
- **Machine Learning**
- **Fuzzy Logic**
- **Rule-Based System**

The system uses AMT sensor features to:
- detect anomaly severity
- recommend a corrective action
- compare performance using inference time, classification quality and false positive / false negative behaviour

## Files Description

### `Question1.xlsx`
Excel Solver model for the CSP-based patrol scheduling problem.

### `Question2Automotive.rdf`
RDF/OWL ontology file representing the automotive knowledge model.

### `Question3.py`
Python implementation of IDDFS and Best First Search with multiple heuristics for maze pathfinding.

### `Question4.py`
Python implementation of three AMT anomaly detection approaches: Machine Learning, Fuzzy Logic and Rule-Based reasoning.

### `2506755_20240281_Nisitha Nimsara.pdf`
Final coursework report containing explanations, screenshots, outputs, analysis and conclusions for all questions.

## How to Run

### Question 1
Open `Question1.xlsx` in **Microsoft Excel** and make sure the **Solver Add-in** is enabled.

### Question 2
Open `Question2Automotive.rdf` in a tool such as:
- **Protégé**
- **Apache Jena**
- **Twinkle**

You can use these tools to inspect the ontology and run SPARQL queries.

### Question 3
Run the Python file directly:

```bash
python Question3.py
```

### Question 4
Before running, place the AMT dataset file in the project root as:

```text
AMT_Anomaly_Dataset.csv
```

Then install the required Python packages:

```bash
pip install numpy pandas scikit-learn scikit-fuzzy
```

Run the program:

```bash
python Question4.py
```

## Technologies Used

- Microsoft Excel Solver
- Python
- RDF / OWL
- SPARQL
- scikit-learn
- scikit-fuzzy
- NumPy
- Pandas

## Key Learning Areas

This repository demonstrates practical work in:
- constraint modelling and optimisation
- knowledge representation
- ontology design and querying
- uninformed and informed search
- heuristic evaluation
- anomaly detection and AI model comparison

## Academic Note

This repository is shared for **learning and portfolio purposes**. Please do not copy, reuse, or submit this work as your own in any academic setting. Use it only as a reference while following your institution’s academic integrity guidelines.


