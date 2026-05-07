# Source Code - NIR Package

Core Python package for NIR spectral analysis ML pipeline.

## Structure

```
src/deep_nir/
├── preprocessing/   # Data preprocessing and cleaning modules
├── features/        # Feature engineering and selection
├── models/          # Model architectures and training logic
├── explainability/  # Model interpretability and explainability
└── utils/           # Utility functions and helpers
```

## Module Descriptions

### preprocessing/
Data preprocessing pipeline components:
- Data loading and validation
- Missing value imputation
- Outlier detection and handling
- Data normalization and standardization
- Spectral preprocessing (baseline correction, smoothing, etc.)

### features/
Feature engineering for NIR spectral data:
- Feature extraction from spectra
- Feature selection algorithms
- Dimensionality reduction (PCA, t-SNE)
- Domain-specific transformations

### models/
ML model implementations:
- Model architectures (framework-agnostic base classes)
- Training and validation logic
- Model persistence and loading
- Ensemble methods

### explainability/
Model interpretability tools:
- Feature importance analysis
- SHAP/LIME integrations
- Model visualization
- Prediction explanations

### utils/
Shared utilities:
- Logging configuration
- File I/O helpers
- Configuration loaders
- Common validators

## Development Guidelines

1. **Type hints**: Use type annotations for all functions
2. **Docstrings**: Follow NumPy or Google docstring format
3. **Testing**: Write unit tests for all public functions
4. **Framework-agnostic**: Keep core logic independent of specific ML frameworks
5. **Modularity**: Each module should have a single, well-defined responsibility

## Installation

```bash
# Development mode installation
uv pip install -e .

# With specific ML framework (user's choice)
uv pip install -e ".[torch]"  # or tensorflow, sklearn, etc.
```
