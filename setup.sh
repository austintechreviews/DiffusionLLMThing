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
NC='\033[0m' # No Color

# Default values
VENV_DIR="venv"
PYTHON_CMD="python3"
INSTALL_DEV=false
INSTALL_ALL=false

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev         Install development dependencies (pytest, black, ruff)"
            echo "  --all         Install all optional dependencies (wandb, tensorboard, etc.)"
            echo "  --python CMD  Use specified Python command (default: python3)"
            echo "  --venv-dir    Specify virtual environment directory (default: venv)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
elif [ "$INSTALL_DEV" = true ]; then
    echo "  Installing development dependencies..."
    pip install -e ".[dev]" --quiet
else
    echo "  Installing core dependencies..."
    pip install -e "." --quiet
fi

echo -e "${GREEN}  Dependencies installed${NC}"

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import diffusionllm; print(f'  diffusionllm: {diffusionllm.__version__}')"
echo -e "${GREEN}  Verification complete${NC}"

# Run tests if dev mode
if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_ALL" = true ]; then
    echo ""
    echo -e "${YELLOW}Running tests...${NC}"
    python -m pytest tests/ -v --tb=short -q
    echo -e "${GREEN}  Tests passed${NC}"
fi

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To activate the virtual environment:"
echo -e "  ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "To train a model:"
echo -e "  ${YELLOW}python scripts/train.py --config configs/default.yaml${NC}"
echo ""
echo "To generate samples:"
echo -e "  ${YELLOW}python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt${NC}"
echo ""
echo "To run tests:"
echo -e "  ${YELLOW}pytest tests/ -v${NC}"
echo ""

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
# Helper script to activate the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo "Virtual environment activated!"
EOF
chmod +x activate.sh

echo -e "${YELLOW}Tip: Run 'source activate.sh' to quickly activate the venv${NC}"
echo ""
