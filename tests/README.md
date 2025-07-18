# Running Tests

## Quick Test Run
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for workflows  
- `tests/conftest.py` - Shared test fixtures
- `tests/utils.py` - Test data generation utilities

## Test Data
Generate synthetic test data:
```python
from tests.utils import generate_test_dataset
generate_test_dataset('test_data', num_images=50)
```
