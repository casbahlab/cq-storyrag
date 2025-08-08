VENV_DIR := wemb
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Create virtual environment
$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)

# Install dependencies
install: $(VENV_DIR)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Virtual environment created at './$(VENV_DIR)'"
	@echo "To activate it, run:"
	@echo "   source $(VENV_DIR)/bin/activate"
	@echo ""

# Clean environment and output
clean:
	rm -rf $(VENV_DIR) output/*.csv
