#!/bin/bash
#
# Run script for Discrete Diffusion Language Model
# Provides interactive menu and command execution with venv
#
# Usage:
#   ./run.sh                              # Interactive menu
#   ./run.sh train --test                 # Run training test
#   ./run.sh chat                         # Start chat
#   ./run.sh server --port 8000           # Start API server
#   ./run.sh python script.py             # Run arbitrary Python
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if venv exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}"
        echo "Run './setup.sh' first to create it."
        echo ""
        read -p "Run setup now? [Y/n] " run_setup
        if [[ ! "$run_setup" =~ ^[Nn]$ ]]; then
            ./setup.sh --interactive
        else
            exit 1
        fi
    fi
}

# Activate venv
activate() {
    source "$VENV_DIR/bin/activate"
}

# Print main menu
print_menu() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Discrete Diffusion LM - Run Menu${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${CYAN}Training:${NC}"
    echo "  1) Quick test (tiny model, 100 steps)"
    echo "  2) Train with preset (tiny/small/base)"
    echo "  3) Resume from checkpoint"
    echo ""
    echo -e "${CYAN}Generation:${NC}"
    echo "  4) Interactive chat"
    echo "  5) Batch sampling"
    echo "  6) Start API server"
    echo ""
    echo -e "${CYAN}Data:${NC}"
    echo "  7) Prepare data"
    echo "  8) Evaluate model"
    echo ""
    echo -e "${CYAN}System:${NC}"
    echo "  9) Check installation"
    echo "  10) Run tests"
    echo "  11) Python REPL"
    echo ""
    echo "  0) Exit"
    echo ""
}

# Check installation
check_installation() {
    echo -e "${YELLOW}Checking installation...${NC}"
    echo ""
    
    activate
    
    echo -e "${BLUE}Core:${NC}"
    $PYTHON -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo -e "  ${RED}✗ PyTorch${NC}"
    $PYTHON -c "import diffusionllm; print(f'  ✓ diffusionllm: {diffusionllm.__version__}')" 2>/dev/null || echo -e "  ${RED}✗ diffusionllm${NC}"
    
    echo -e "\n${BLUE}Optional:${NC}"
    $PYTHON -c "import tokenizers; print(f'  ✓ tokenizers: {tokenizers.__version__}')" 2>/dev/null && echo -e "  ${GREEN}(required for prepare_data.py)${NC}" || echo -e "  ${YELLOW}⚠ tokenizers (pip install tokenizers)${NC}"
    $PYTHON -c "import fastapi; print(f'  ✓ fastapi: {fastapi.__version__}')" 2>/dev/null && echo -e "  ${GREEN}(required for server.py)${NC}" || echo -e "  ${YELLOW}⚠ fastapi (pip install fastapi uvicorn)${NC}"
    $PYTHON -c "import wandb; print(f'  ✓ wandb: {wandb.__version__}')" 2>/dev/null && echo -e "  ${GREEN}(required for --use-wandb)${NC}" || echo -e "  ${YELLOW}⚠ wandb (pip install wandb)${NC}"
    
    echo -e "\n${BLUE}Data:${NC}"
    if [ -d "data/processed" ]; then
        echo -e "  ✓ Processed data found"
        ls -la data/processed/
    else
        echo -e "  ${YELLOW}⚠ No processed data (run: ./run.sh prepare)${NC}"
    fi
    
    echo -e "\n${BLUE}Checkpoints:${NC}"
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints 2>/dev/null)" ]; then
        echo -e "  ✓ Checkpoints found:"
        ls -1 checkpoints/*.pt 2>/dev/null | head -5
    else
        echo -e "  ${YELLOW}⚠ No checkpoints (run: ./run.sh train)${NC}"
    fi
    
    echo ""
}

# Run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    activate
    $PYTHON -m pytest tests/ -v --tb=short "$@"
}

# Interactive chat
run_chat() {
    echo -e "${YELLOW}Starting interactive chat...${NC}"
    echo ""
    
    # Find latest checkpoint
    CHECKPOINT=""
    if [ -d "checkpoints" ]; then
        CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo -e "${RED}No checkpoint found!${NC}"
        echo "Train a model first: ./run.sh train"
        return
    fi
    
    echo -e "Using checkpoint: ${GREEN}$CHECKPOINT${NC}"
    echo ""
    
    activate
    $PYTHON scripts/chat.py --checkpoint "$CHECKPOINT" "$@"
}

# Start server
run_server() {
    echo -e "${YELLOW}Starting API server...${NC}"
    echo ""
    
    # Find latest checkpoint
    CHECKPOINT=""
    if [ -d "checkpoints" ]; then
        CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo -e "${RED}No checkpoint found!${NC}"
        echo "Train a model first: ./run.sh train"
        return
    fi
    
    echo -e "Using checkpoint: ${GREEN}$CHECKPOINT${NC}"
    echo ""
    
    activate
    $PYTHON scripts/server.py --checkpoint "$CHECKPOINT" "$@"
}

# Prepare data
run_prepare() {
    echo -e "${YELLOW}Data Preparation${NC}"
    echo ""
    
    # Find input directory
    INPUT_DIR=""
    if [ -d "data/raw" ]; then
        INPUT_DIR="data/raw"
    elif [ -d "data" ]; then
        INPUT_DIR="data"
    fi
    
    if [ -n "$INPUT_DIR" ]; then
        echo -e "Found data directory: ${GREEN}$INPUT_DIR${NC}"
    else
        echo -e "${YELLOW}No data directory found.${NC}"
        echo "Create data/raw/ and add your text files there."
        echo ""
        read -p "Enter input directory path: " INPUT_DIR
    fi
    
    echo ""
    read -p "Output directory (default: data/processed): " OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-data/processed}
    
    echo ""
    read -p "Vocabulary size (default: 32000): " VOCAB_SIZE
    VOCAB_SIZE=${VOCAB_SIZE:-32000}
    
    echo ""
    activate
    $PYTHON scripts/prepare_data.py --input "$INPUT_DIR" --output "$OUTPUT_DIR" --vocab-size "$VOCAB_SIZE"
}

# Train model
run_train() {
    echo -e "${YELLOW}Training${NC}"
    echo ""
    
    # Check for processed data
    if [ ! -d "data/processed" ]; then
        echo -e "${YELLOW}No processed data found.${NC}"
        read -p "Prepare data first? [Y/n] " prepare
        if [[ ! "$prepare" =~ ^[Nn]$ ]]; then
            run_prepare
        fi
        echo ""
    fi
    
    # Select preset
    echo "Select model preset:"
    echo "  1) micro  (~1M params, very small datasets)"
    echo "  2) tiny   (~3M params, for testing)"
    echo "  3) small  (~20M params, quick experiments)"
    echo "  4) base   (~85M params, standard)"
    echo "  5) custom"
    echo ""
    read -p "Choice [1-5] (default: 2): " preset_choice
    
    case ${preset_choice:-2} in
        2) PRESET="tiny" ;;
        3) PRESET="small" ;;
        4) PRESET="base" ;;
        5)
            read -p "Enter preset (micro/tiny/small/base/medium/large/xl): " PRESET
            ;;
        *) PRESET="micro" ;;
    esac
    
    echo ""
    read -p "Max steps (default: 1000 for tiny, 100000 for others): " MAX_STEPS
    if [ -z "$MAX_STEPS" ]; then
        if [ "$PRESET" = "tiny" ]; then
            MAX_STEPS=1000
        else
            MAX_STEPS=100000
        fi
    fi
    
    echo ""
    read -p "Batch size (default: 32): " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-32}
    
    echo ""
    read -p "Use rotary embeddings? [y/N] " USE_ROPE
    ROPE_FLAG=""
    if [[ "$USE_ROPE" =~ ^[Yy]$ ]]; then
        ROPE_FLAG="--use-rotary-embeddings"
    fi
    
    echo ""
    echo -e "${BLUE}Starting training...${NC}"
    echo "  Preset: $PRESET"
    echo "  Max steps: $MAX_STEPS"
    echo "  Batch size: $BATCH_SIZE"
    [ -n "$ROPE_FLAG" ] && echo "  Rotary embeddings: yes"
    echo ""
    
    activate
    $PYTHON scripts/train.py \
        --data-dir data/processed \
        --model-preset "$PRESET" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        $ROPE_FLAG \
        "$@"
}

# Resume training
run_resume() {
    echo -e "${YELLOW}Resume Training${NC}"
    echo ""
    
    # Find checkpoints
    if [ ! -d "checkpoints" ] || [ -z "$(ls checkpoints/*.pt 2>/dev/null)" ]; then
        echo -e "${RED}No checkpoints found!${NC}"
        return
    fi
    
    echo "Available checkpoints:"
    ls -lt checkpoints/*.pt 2>/dev/null | head -10
    echo ""
    
    read -p "Enter checkpoint path: " CHECKPOINT
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo -e "${RED}Checkpoint not found: $CHECKPOINT${NC}"
        return
    fi
    
    echo ""
    echo -e "${BLUE}Resuming from: $CHECKPOINT${NC}"
    echo ""
    
    activate
    $PYTHON scripts/train.py --resume "$CHECKPOINT" "$@"
}

# Batch sampling
run_sample() {
    echo -e "${YELLOW}Batch Sampling${NC}"
    echo ""
    
    # Find latest checkpoint
    CHECKPOINT=""
    if [ -d "checkpoints" ]; then
        CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo -e "${RED}No checkpoint found!${NC}"
        return
    fi
    
    echo -e "Using checkpoint: ${GREEN}$CHECKPOINT${NC}"
    echo ""
    
    read -p "Number of samples (default: 5): " NUM_SAMPLES
    NUM_SAMPLES=${NUM_SAMPLES:-5}
    
    read -p "Max length (default: 128): " MAX_LENGTH
    MAX_LENGTH=${MAX_LENGTH:-128}
    
    read -p "Temperature (default: 1.0): " TEMP
    TEMP=${TEMP:-1.0}
    
    echo ""
    activate
    $PYTHON scripts/sample.py \
        --checkpoint "$CHECKPOINT" \
        --num-samples "$NUM_SAMPLES" \
        --max-length "$MAX_LENGTH" \
        --temperature "$TEMP" \
        "$@"
}

# Evaluate model
run_evaluate() {
    echo -e "${YELLOW}Evaluation${NC}"
    echo ""
    
    # Find latest checkpoint
    CHECKPOINT=""
    if [ -d "checkpoints" ]; then
        CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo -e "${RED}No checkpoint found!${NC}"
        return
    fi
    
    echo -e "Using checkpoint: ${GREEN}$CHECKPOINT${NC}"
    echo ""
    
    # Check for test data
    if [ ! -f "data/processed/test.jsonl" ]; then
        echo -e "${YELLOW}No test data found at data/processed/test.jsonl${NC}"
        return
    fi
    
    activate
    $PYTHON scripts/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data-path data/processed/test.jsonl \
        --eval-timesteps \
        "$@"
}

# Python REPL
run_python() {
    activate
    $PYTHON "$@"
}

# Main function
main() {
    check_venv
    
    # If arguments provided, run directly
    if [ $# -gt 0 ]; then
        case "$1" in
            train)
                shift
                run_train "$@"
                ;;
            chat)
                shift
                run_chat "$@"
                ;;
            server)
                shift
                run_server "$@"
                ;;
            sample)
                shift
                run_sample "$@"
                ;;
            prepare|data)
                shift
                run_prepare "$@"
                ;;
            resume)
                shift
                run_resume "$@"
                ;;
            eval|evaluate)
                shift
                run_evaluate "$@"
                ;;
            test)
                shift
                run_tests "$@"
                ;;
            check)
                check_installation
                ;;
            python|py)
                shift
                run_python "$@"
                ;;
            *)
                # Pass through to python
                run_python "$@"
                ;;
        esac
        return
    fi
    
    # Interactive menu
    while true; do
        print_menu
        read -p "Select option [0-11]: " choice
        
        case $choice in
            1) run_train --test ;;
            2) run_train ;;
            3) run_resume ;;
            4) run_chat ;;
            5) run_sample ;;
            6) run_server ;;
            7) run_prepare ;;
            8) run_evaluate ;;
            9) check_installation ;;
            10) run_tests ;;
            11)
                activate
                echo -e "${BLUE}Python REPL${NC}"
                echo "Type 'exit()' to quit"
                echo ""
                $PYTHON
                ;;
            0)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

main "$@"
