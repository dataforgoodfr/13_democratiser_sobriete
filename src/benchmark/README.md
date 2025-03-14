# **Retrieval Evaluation Pipeline**

This project provides a complete pipeline for **synthetic query generation**, **retrieval evaluation**, and **performance visualization**.  

## **Features**
### 🔹 **1. Generate a Set of Questions**
Automatically generate a set of synthetic queries based on a given configuration file.

```bash
python ./query_generation/generate_synthetic_queries.py --config_path ./config_generation/query_generation.yaml
```

- **Input**: Configuration file (`query_generation.yaml`) containing generation parameters.  
- **Output**: A dataset of generated queries ready for retrieval evaluation.

---

### 🔹 **2. Evaluate Retrieval Performance**
Run a benchmark evaluation of the retrieval system to assess its effectiveness.

```bash
python ./retrieval_evaluation/benchmark_retrieval.py --config_path ./config_retrieval/config_retrieval.yaml
```

- **Input**: Configuration file (`config_retrieval.yaml`) specifying retrieval settings.  
- **Output**: Evaluation results saved as a CSV file in the `outputs/` directory.

---

### 🔹 **3. Visualize Performance Metrics**
Generate and save visualizations of retrieval performance metrics.

```bash
python ./utils/visualization.py --eval_file_name ./outputs/evaluation_test.csv --limit_k 10 --save --output_file ./outputs/metrics_viz.png
```

- **Parameters**:
  - `--eval_file_name`: Path to the evaluation results CSV file.
  - `--limit_k`: Number of top results to visualize (default: 10).
  - `--save`: Flag to save the generated plot.
  - `--output_file`: Path to save the visualization image.

- **Output**: A graphical representation of evaluation metrics saved as an image.

---

## **Next Steps**
- **Optimize Query Generation**: Tune parameters in `query_generation.yaml` for better results.
- **Enhance Retrieval Models**: Experiment with different retrieval configurations.
- **Expand Visualizations**: Customize plots for deeper analysis.

---
