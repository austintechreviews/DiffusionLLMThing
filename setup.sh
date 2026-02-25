#!/bin/bash
#
# Setup script for Discrete Diffusion Language Model
# Creates a virtual environment and installs all dependencies
#
# Usage:
#   ./setup.sh                    # Default setup
#   ./setup.sh --dev              # Include development dependencies
#   ./setup.sh --all              # Include all optional dependencies
#   ./setup.sh --python python3   # Specify Python version
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VENV_DIR="venv"
PYTHON_CMD="python3"
INSTALL_DEV=false
INSTALL_ALL=false
INTERACTIVE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --interactive   Interactive setup wizard"
            echo "  --dev               Install development dependencies (pytest, black, ruff)"
            echo "  --all               Install all optional dependencies (wandb, tensorboard, etc.)"
            echo "  --python CMD        Use specified Python command (default: python3)"
            echo "  --venv-dir          Specify virtual environment directory (default: venv)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Interactive mode
if [ "$INTERACTIVE" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Discrete Diffusion LM - Setup Wizard${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Python selection
    echo -e "${YELLOW}Step 1: Python Version${NC}"
    echo "Available Python versions:"
    python_versions=()
    for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd --version 2>&1 | cut -d' ' -f2)
            python_versions+=("$cmd:$version")
            echo "  - $cmd ($version)"
        fi
    done
    
    if [ ${#python_versions[@]} -eq 0 ]; then
        echo -e "${RED}No Python installation found!${NC}"
        exit 1
    fi
    
    echo ""
    read -p "Select Python version (press Enter for default): " python_choice
    
    if [ -n "$python_choice" ]; then
        for pv in "${python_versions[@]}"; do
            cmd="${pv%%:*}"
            if [ "$python_choice" = "$cmd" ]; then
                PYTHON_CMD="$cmd"
                break
            fi
        done
    fi
    
    echo -e "${GREEN}Using: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1 | cut -d' ' -f2))${NC}"
    echo ""
    
    # Installation type
    echo -e "${YELLOW}Step 2: Installation Type${NC}"
    echo "  1) Core only (minimal)"
    echo "  2) Development (includes pytest, black, ruff)"
    echo "  3) Full (includes wandb, tensorboard, fastapi)"
    echo ""
    read -p "Select installation type [1-3] (default: 2): " install_choice
    
    case ${install_choice:-2} in
        1)
            echo -e "${GREEN}Installing core dependencies only${NC}"
            ;;
        3)
            INSTALL_ALL=true
            echo -e "${GREEN}Installing all dependencies${NC}"
            ;;
        *)
            INSTALL_DEV=true
            echo -e "${GREEN}Installing development dependencies${NC}"
            ;;
    esac
    echo ""
    
    # venv location
    echo -e "${YELLOW}Step 3: Virtual Environment Location${NC}"
    read -p "Enter venv directory name (default: venv): " venv_choice
    if [ -n "$venv_choice" ]; then
        VENV_DIR="$venv_choice"
    fi
    echo -e "${GREEN}Using: $VENV_DIR${NC}"
    echo ""
    
    # Confirmation
    echo -e "${YELLOW}Summary:${NC}"
    echo "  Python: $PYTHON_CMD"
    echo "  venv: $VENV_DIR"
    if [ "$INSTALL_ALL" = true ]; then
        echo "  Install: Full (all dependencies)"
    elif [ "$INSTALL_DEV" = true ]; then
        echo "  Install: Development"
    else
        echo "  Install: Core only"
    fi
    echo ""
    read -p "Continue? [Y/n] " confirm
    
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Discrete Diffusion LM - Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_CMD not found${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "  Python version: $PYTHON_VERSION"

# Check minimum Python version (3.9)
REQUIRED_VERSION="3.9"
if [[ $(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher required${NC}"
    exit 1
fi

# Check for venv module
echo -e "${YELLOW}Checking for venv module...${NC}"
if ! $PYTHON_CMD -m venv --help &> /dev/null; then
    echo -e "${RED}Error: venv module not available${NC}"
    echo "  Install it with: sudo apt install python3-venv  (Ubuntu/Debian)"
    echo "                   sudo pacman -S python-venv     (Arch)"
    exit 1
fi
echo "  venv module: OK"

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment in '$VENV_DIR'...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}  Virtual environment already exists, removing...${NC}"
    rm -rf "$VENV_DIR"
fi

$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}  Virtual environment created${NC}"

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  Activated${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}  pip upgraded${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"

if [ "$INSTALL_ALL" = true ]; then
    echo "  Installing all optional dependencies..."
    pip install -e ".[all]" --quiet
    echo -e "${GREEN}  ✓ All dependencies installed${NC}"
elif [ "$INSTALL_DEV" = true ]; then
    echo "  Installing development dependencies..."
    pip install -e ".[dev]" --quiet
    echo -e "${GREEN}  ✓ Development dependencies installed${NC}"
else
    echo "  Installing core dependencies..."
    pip install -e "." --quiet
    echo -e "${GREEN}  ✓ Core dependencies installed${NC}"
fi

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')"
python -c "import diffusionllm; print(f'  ✓ diffusionllm: {diffusionllm.__version__}')"

# Check optional dependencies
if python -c "import tokenizers" 2>/dev/null; then
    python -c "import tokenizers; print(f'  ✓ tokenizers: {tokenizers.__version__}')"
else
    echo -e "  ⚠ tokenizers: not installed (run: pip install tokenizers)"
fi

if python -c "import fastapi" 2>/dev/null; then
    python -c "import fastapi; print(f'  ✓ fastapi: {fastapi.__version__}')"
else
    echo -e "  ⚠ fastapi: not installed (run: pip install fastapi uvicorn)"
fi

if python -c "import wandb" 2>/dev/null; then
    python -c "import wandb; print(f'  ✓ wandb: {wandb.__version__}')"
else
    echo -e "  ⚠ wandb: not installed (run: pip install wandb)"
fi

echo -e "${GREEN}  Verification complete${NC}"

# Run tests if dev mode
if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_ALL" = true ]; then
    echo ""
    echo -e "${YELLOW}Running tests...${NC}"
    if python -m pytest tests/ -v --tb=short -q 2>/dev/null; then
        echo -e "${GREEN}  ✓ All tests passed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Some tests failed (this is okay for development)${NC}"
    fi
fi

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "2. Prepare your data:"
echo -e "   ${YELLOW}python scripts/prepare_data.py --input data/raw/ --output data/processed${NC}"
echo ""
echo "3. Train a model:"
echo -e "   ${YELLOW}python scripts/train.py --data-dir data/processed --model-preset base${NC}"
echo ""
echo "4. Generate text:"
echo -e "   ${YELLOW}python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt${NC}"
echo ""
echo "5. Start API server:"
echo -e "   ${YELLOW}python scripts/server.py --checkpoint checkpoints/checkpoint_final.pt --port 8000${NC}"
echo ""

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
# Helper script to activate the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo -e "\033[0;32m✓ Virtual environment activated!\033[0m"
echo ""
echo "Available commands:"
echo "  diffusion-train     - Training script"
echo "  diffusion-sample    - Sampling script"
echo "  diffusion-chat      - Interactive chat"
echo "  diffusion-server    - API server"
echo "  diffusion-prepare-data - Data preparation"
echo ""
EOF
chmod +x activate.sh

echo -e "${YELLOW}Tip: Run 'source activate.sh' to quickly activate the venv${NC}"
echo ""

# Quick start menu
if [ "$INTERACTIVE" = true ]; then
    echo -e "${YELLOW}Would you like to run a quick test now?${NC}"
    read -p "Run test training? [y/N] " run_test
    
    if [[ "$run_test" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${BLUE}Running quick test (tiny model, 100 steps)...${NC}"
        python scripts/train.py --test
        echo ""
        echo -e "${GREEN}Test complete! Check checkpoints_test/ for outputs${NC}"
    fi
fi
