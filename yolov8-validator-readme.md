# YOLOv8 Dataset Validator

A comprehensive tool for analyzing, validating, and generating detailed statistics for YOLOv8 datasets.

## Overview

The YOLOv8 Dataset Validator is designed to ensure data quality and provide insightful statistics about your object detection datasets. It performs thorough validation of both images and annotations, identifying common issues and generating comprehensive reports that can help improve model training outcomes.

## Features

- **Complete Dataset Analysis**: Validates images and labels across training, validation, and test splits
- **Class Distribution Statistics**: Visualizes class balance and identifies potential imbalance issues
- **Bounding Box Analysis**: Analyzes size, position, aspect ratio, and area distributions
- **Image Statistics**: Examines image dimensions, aspect ratios, and formats
- **Issue Detection**: Identifies problematic annotations, corrupt images, and missing labels
- **Interactive HTML Report**: Generates a user-friendly visual report of all findings
- **Sample Visualizations**: Creates annotated samples from your dataset for visual inspection
- **CI/CD Integration**: Supports Jenkins and other CI/CD pipelines with machine-readable outputs
- **Class Name Support**: Properly handles class names from data.yaml files

## Installation

### Prerequisites

- Python 3.7+
- Required packages:
  - numpy
  - opencv-python
  - matplotlib
  - pandas
  - seaborn
  - pillow
  - pyyaml
  - tqdm

### Installation

```bash
# Clone the repository (if applicable)
git clone https://github.com/yourusername/yolov8-dataset-validator.git
cd yolov8-dataset-validator

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

Run the validator on your dataset:

```bash
python data_validator.py --dataset_path /path/to/dataset --yaml_path /path/to/data.yaml
```

### Command Line Arguments

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--dataset_path` | `-d` | Path to YOLOv8 dataset directory | Yes |
| `--yaml_path` | `-y` | Path to dataset YAML file with class names | No |
| `--output_dir` | `-o` | Directory to save validation results | No |

## Advanced Features

### CI/CD Integration

The validator supports integration with CI/CD pipelines like Jenkins:

```bash
python data_validator.py --dataset_path /path/to/dataset --yaml_path /path/to/data.yaml \
  --ci_mode --json_report --fail_on_issues --issue_threshold 10
```

| Argument | Description |
|----------|-------------|
| `--ci_mode` | Run with minimal output for CI environments |
| `--json_report` | Generate machine-readable JSON report |
| `--fail_on_issues` | Exit with error code if issues exceed threshold |
| `--issue_threshold` | Number of issues to tolerate before failing |

### Example Jenkinsfile

```groovy
pipeline {
    agent any
    
    parameters {
        string(name: 'DATASET_PATH', defaultValue: '/path/to/dataset', description: 'Path to YOLOv8 dataset')
        string(name: 'YAML_PATH', defaultValue: '/path/to/data.yaml', description: 'Path to dataset YAML file')
    }
    
    stages {
        stage('Validate Dataset') {
            steps {
                sh """
                python data_validator.py \
                    --dataset_path ${params.DATASET_PATH} \
                    --yaml_path ${params.YAML_PATH} \
                    --output_dir validation_results \
                    --ci_mode \
                    --json_report \
                    --fail_on_issues
                """
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'validation_results/**/*.*', allowEmptyArchive: true
        }
    }
}
```

## Output

The validator generates a comprehensive analysis in your specified output directory:

- **HTML Report**: Interactive report with visualizations and statistics
- **Summary Reports**: Both JSON and text formats with key statistics
- **Visualizations**: Charts for class distribution, bounding box statistics, etc.
- **Issue List**: CSV file with all detected problems and their locations
- **Sample Images**: Annotated samples from your dataset

## Examples

### Basic Validation
```bash
python data_validator.py -d ./my_dataset -y ./my_dataset/data.yaml
```

### Custom Output Directory
```bash
python data_validator.py -d ./my_dataset -y ./my_dataset/data.yaml -o ./validation_report
```

### Automated Pipeline with Strict Validation
```bash
python data_validator.py -d ./my_dataset -y ./my_dataset/data.yaml --ci_mode --fail_on_issues --issue_threshold 5
```

## Expected Dataset Structure

The validator expects a standard YOLOv8 dataset structure:

```
dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/ (or valid/)
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
```

## Class Names from YAML

The validator supports loading class names from your `data.yaml` file:

```yaml
# Example data.yaml
nc: 12  # number of classes
names: ['big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck', 'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
```

These class names will be used in all visualizations and reports instead of generic "class_0", "class_1", etc.

## Jenkins
```
pipeline {
    agent any
    
    parameters {
        string(name: 'DATASET_PATH', defaultValue: '/path/to/dataset', description: 'Path to YOLOv8 dataset')
        string(name: 'YAML_PATH', defaultValue: '/path/to/data.yaml', description: 'Path to dataset YAML file')
        booleanParam(name: 'FAIL_ON_ISSUES', defaultValue: true, description: 'Fail build if issues are found')
        string(name: 'ISSUE_THRESHOLD', defaultValue: '10', description: 'Number of issues to tolerate before failing')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Validate Dataset') {
            steps {
                script {
                    try {
                        sh """
                        python data_validator.py \\
                            --dataset_path ${params.DATASET_PATH} \\
                            --yaml_path ${params.YAML_PATH} \\
                            --output_dir validation_results \\
                            --ci_mode \\
                            --json_report \\
                            ${params.FAIL_ON_ISSUES ? '--fail_on_issues' : ''} \\
                            --issue_threshold ${params.ISSUE_THRESHOLD}
                        """
                    } catch (Exception e) {
                        currentBuild.result = 'FAILURE'
                        error "Dataset validation failed: ${e.message}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'validation_results/**/*.*', allowEmptyArchive: true
            
            script {
                if (fileExists('validation_results/ci_report.json')) {
                    def reportJson = readJSON file: 'validation_results/ci_report.json'
                    echo "Dataset Validation Summary:"
                    echo "- Total Images: ${reportJson.total_images}"
                    echo "- Valid Images: ${reportJson.valid_images}"
                    echo "- Issues Found: ${reportJson.issues_count}"
                    
                    if (reportJson.status == 'warning') {
                        currentBuild.description = "⚠️ Issues: ${reportJson.issues_count}"
                    } else {
                        currentBuild.description = "✅ Clean"
                    }
                }
            }
        }
        
        success {
            echo "Dataset validation completed successfully"
        }
        
        failure {
            echo "Dataset validation failed"
        }
    }
}
```

### Dockerfile
```
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_validator.py .

ENTRYPOINT ["python", "data_validator.py"]
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all packages in requirements.txt are installed
2. **Invalid dataset structure**: Check that your dataset follows the expected structure
3. **Path issues**: Use absolute paths if facing issues with relative paths
4. **Memory errors**: For very large datasets, use an environment with sufficient RAM

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
