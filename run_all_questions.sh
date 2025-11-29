#!/bin/bash

# Master script to run all CENG543 Midterm questions
# Usage: ./run_all_questions.sh [q1|q2|q3|q4|q5|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if Python environment is properly set up
check_environment() {
    print_header "Checking Python Environment"

    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_success "Conda environment detected: $CONDA_DEFAULT_ENV"
    else
        print_warning "No virtual environment detected!"
        echo "It's recommended to use a virtual environment."
        echo ""
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "Please set up the environment first:"
            echo "  ${BLUE}./setup_environment.sh${NC}"
            exit 1
        fi
    fi

    # Check if required packages are installed
    print_info "Checking required packages..."

    local missing_packages=()

    python3 -c "import torch" 2>/dev/null || missing_packages+=("torch")
    python3 -c "import transformers" 2>/dev/null || missing_packages+=("transformers")
    python3 -c "import datasets" 2>/dev/null || missing_packages+=("datasets")
    python3 -c "import numpy" 2>/dev/null || missing_packages+=("numpy")
    python3 -c "import sklearn" 2>/dev/null || missing_packages+=("scikit-learn")

    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        echo ""
        echo "Please install dependencies first:"
        echo "  ${BLUE}./setup_environment.sh${NC}"
        echo ""
        echo "Or manually install requirements:"
        echo "  ${BLUE}pip install -r requirements.txt${NC}"
        exit 1
    fi

    print_success "All required packages are installed"

    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2)
    print_info "Python version: $python_version"

    # Check PyTorch and CUDA availability
    local cuda_available=$(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "No")
    if [ "$cuda_available" = "Yes" ]; then
        local cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        print_success "CUDA available: $cuda_version"
    else
        print_warning "CUDA not available - will use CPU (training will be slower)"
    fi

    echo ""
}

# Function to run Question 1
run_q1() {
    print_header "Running Question 1"
    cd ceng543_m_q1
    if [ -f "run_all_q1.sh" ]; then
        chmod +x run_all_q1.sh
        ./run_all_q1.sh
        print_success "Question 1 completed"
    else
        print_error "run_all_q1.sh not found"
        return 1
    fi
    cd ..
}

# Function to run Question 2
run_q2() {
    print_header "Running Question 2"
    cd ceng543_m_q2
    if [ -f "run_all.sh" ]; then
        chmod +x run_all.sh
        ./run_all.sh
        print_success "Question 2 completed"
    else
        print_error "run_all.sh not found"
        return 1
    fi
    cd ..
}

# Function to run Question 3
run_q3() {
    print_header "Running Question 3"
    cd ceng543_m_q3
    if [ -f "run_all_q3_experiments.sh" ]; then
        chmod +x run_all_q3_experiments.sh
        ./run_all_q3_experiments.sh
        print_success "Question 3 completed"
    else
        print_error "run_all_q3_experiments.sh not found"
        return 1
    fi
    cd ..
}

# Function to run Question 4
run_q4() {
    print_header "Running Question 4"
    cd ceng543_m_q4
    if [ -f "run_all_q4.sh" ]; then
        chmod +x run_all_q4.sh
        ./run_all_q4.sh
        print_success "Question 4 completed"
    else
        print_error "run_all_q4.sh not found"
        return 1
    fi
    cd ..
}

# Function to run Question 5
run_q5() {
    print_header "Running Question 5"
    cd ceng543_m_q5
    if [ -f "run_all_q5.sh" ]; then
        chmod +x run_all_q5.sh
        ./run_all_q5.sh
        print_success "Question 5 completed"
    else
        print_error "run_all_q5.sh not found"
        return 1
    fi
    cd ..
}

# Main execution
main() {
    local start_time=$(date +%s)

    print_header "CENG543 Midterm - Master Runner"
    print_info "Starting at: $(date)"

    # Check environment before running
    check_environment

    case "${1:-all}" in
        q1)
            run_q1
            ;;
        q2)
            run_q2
            ;;
        q3)
            run_q3
            ;;
        q4)
            run_q4
            ;;
        q5)
            run_q5
            ;;
        all)
            print_info "Running all questions sequentially..."
            run_q1
            run_q2
            run_q3
            run_q4
            run_q5
            ;;
        *)
            print_error "Invalid argument: $1"
            echo "Usage: $0 [q1|q2|q3|q4|q5|all]"
            echo "  q1  - Run only Question 1"
            echo "  q2  - Run only Question 2"
            echo "  q3  - Run only Question 3"
            echo "  q4  - Run only Question 4"
            echo "  q5  - Run only Question 5"
            echo "  all - Run all questions (default)"
            exit 1
            ;;
    esac

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    print_header "All tasks completed!"
    print_success "Total execution time: ${duration} seconds ($(($duration / 60)) minutes)"
    print_info "Finished at: $(date)"
}

# Run main function with all arguments
main "$@"
