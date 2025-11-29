#!/bin/bash

# CENG543 NLP Midterm - Environment Setup Script
# This script helps you set up the Python environment for all questions

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO] $1${NC}"
}

print_step() {
    echo -e "${CYAN}[$1] $2${NC}"
}

# Function to check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if we're in a virtual environment
check_venv() {
    if [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        return 0
    else
        return 1
    fi
}

# Setup using Conda
setup_conda() {
    print_header "Setting up Conda Environment"

    ENV_NAME="ceng543-nlp"

    print_step "1/3" "Checking if environment already exists..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_info "Environment '${ENV_NAME}' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
        else
            print_info "Using existing environment. Updating packages..."
            conda env update -f environment.yml --prune
            print_success "Environment updated successfully!"
            echo -e "\n${GREEN}To activate the environment, run:${NC}"
            echo -e "${CYAN}conda activate ${ENV_NAME}${NC}"
            return 0
        fi
    fi

    print_step "2/3" "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    print_step "3/3" "Verifying installation..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
    python -c "import torch; import transformers; import datasets; print('[SUCCESS] All core packages imported successfully!')"

    print_success "Conda environment setup complete!"
    echo -e "\n${GREEN}To activate the environment, run:${NC}"
    echo -e "${CYAN}conda activate ${ENV_NAME}${NC}"
}

# Setup using pip (virtualenv)
setup_pip() {
    print_header "Setting up Python Virtual Environment"

    VENV_DIR="venv"

    print_step "1/4" "Checking Python version..."
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.8"

    if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
        print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python version: $PYTHON_VERSION"

    print_step "2/4" "Creating virtual environment..."
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
        fi
    else
        python3 -m venv "$VENV_DIR"
    fi

    print_step "3/4" "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    print_step "4/4" "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt

    print_success "Virtual environment setup complete!"
    echo -e "\n${GREEN}To activate the environment, run:${NC}"
    echo -e "${CYAN}source venv/bin/activate${NC}"
}

# Download required data/models
download_data() {
    print_header "Downloading Required Data & Models"

    print_info "Checking for required downloads..."

    # Check if GloVe embeddings are needed
    if [ ! -f "glove.6B.100d.txt" ]; then
        print_step "1/1" "GloVe embeddings will be downloaded when needed by the scripts"
        print_info "The training scripts will automatically download required datasets from HuggingFace"
    else
        print_success "GloVe embeddings already present"
    fi

    print_success "Data check complete"
}

# Main menu
main() {
    print_header "CENG543 NLP Midterm - Environment Setup"

    echo "This script will help you set up the Python environment for all questions."
    echo ""
    echo "Choose your preferred setup method:"
    echo ""
    echo "  1) Conda (Recommended - Better dependency management)"
    echo "  2) pip + virtualenv (Simpler, no conda required)"
    echo "  3) Skip environment setup (I already have everything installed)"
    echo "  4) Exit"
    echo ""

    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            if check_conda; then
                setup_conda
                download_data
            else
                print_error "Conda is not installed!"
                echo "Please install Miniconda or Anaconda first:"
                echo "https://docs.conda.io/en/latest/miniconda.html"
                exit 1
            fi
            ;;
        2)
            setup_pip
            download_data
            ;;
        3)
            print_info "Skipping environment setup..."
            download_data
            ;;
        4)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice!"
            exit 1
            ;;
    esac

    print_header "Setup Complete!"
    echo -e "${GREEN}You're ready to run the experiments!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment (see instructions above)"
    echo "  2. Run all experiments: ${CYAN}./run_all_questions.sh${NC}"
    echo "  3. Or run individual questions: ${CYAN}./run_all_questions.sh q1${NC}"
    echo ""
}

# Run main function
main
