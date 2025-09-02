#!/bin/bash

# Mathematical Analysis Documentation Compilation Script
# This script compiles the LaTeX mathematical analysis documentation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_DOC="main_document"
BUILD_DIR="build"
BIB_FILE="references"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_status "Checking LaTeX dependencies..."
    
    if ! command_exists pdflatex; then
        print_error "pdflatex not found. Please install a LaTeX distribution."
        echo
        echo "Installation instructions:"
        echo "  macOS:     brew install --cask mactex"
        echo "  Ubuntu:    sudo apt-get install texlive-full"
        echo "  CentOS:    sudo yum install texlive-scheme-full"
        exit 1
    fi
    
    if ! command_exists bibtex; then
        print_error "bibtex not found. Please install BibTeX."
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

# Create build directory
setup_build_dir() {
    print_status "Setting up build directory..."
    mkdir -p "$BUILD_DIR"
    print_success "Build directory created: $BUILD_DIR"
}

# Compile document
compile_document() {
    print_status "Compiling mathematical analysis documentation..."
    
    # First LaTeX pass
    print_status "Running first LaTeX pass..."
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_DOC.tex" > "$BUILD_DIR/latex_pass1.log" 2>&1
    if [ $? -ne 0 ]; then
        print_error "First LaTeX pass failed. Check $BUILD_DIR/latex_pass1.log"
        tail -20 "$BUILD_DIR/latex_pass1.log"
        exit 1
    fi
    
    # Process bibliography
    print_status "Processing bibliography..."
    cd "$BUILD_DIR"
    bibtex "$MAIN_DOC" > "bibtex.log" 2>&1
    if [ $? -ne 0 ]; then
        print_warning "BibTeX processing had warnings. Check $BUILD_DIR/bibtex.log"
    fi
    cd ..
    
    # Second LaTeX pass
    print_status "Running second LaTeX pass..."
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_DOC.tex" > "$BUILD_DIR/latex_pass2.log" 2>&1
    if [ $? -ne 0 ]; then
        print_error "Second LaTeX pass failed. Check $BUILD_DIR/latex_pass2.log"
        tail -20 "$BUILD_DIR/latex_pass2.log"
        exit 1
    fi
    
    # Third LaTeX pass (final)
    print_status "Running final LaTeX pass..."
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_DOC.tex" > "$BUILD_DIR/latex_pass3.log" 2>&1
    if [ $? -ne 0 ]; then
        print_error "Final LaTeX pass failed. Check $BUILD_DIR/latex_pass3.log"
        tail -20 "$BUILD_DIR/latex_pass3.log"
        exit 1
    fi
    
    print_success "Document compilation completed successfully!"
}

# Generate document statistics
generate_stats() {
    print_status "Generating document statistics..."
    
    if [ -f "$BUILD_DIR/$MAIN_DOC.pdf" ]; then
        PDF_SIZE=$(du -h "$BUILD_DIR/$MAIN_DOC.pdf" | cut -f1)
        print_success "PDF generated: $BUILD_DIR/$MAIN_DOC.pdf ($PDF_SIZE)"
        
        # Count pages if pdfinfo is available
        if command_exists pdfinfo; then
            PAGES=$(pdfinfo "$BUILD_DIR/$MAIN_DOC.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}')
            if [ -n "$PAGES" ]; then
                print_success "Document pages: $PAGES"
            fi
        fi
        
        # Count approximate words
        WORD_COUNT=$(find . -name "*.tex" -exec cat {} \; | wc -w | tr -d ' ')
        print_success "Approximate word count: $WORD_COUNT"
        
        echo
        echo "Document structure:"
        echo "  ├── Convergence Proofs (Gaussian Process, MPC, Bayesian RL)"
        echo "  ├── Stability Analysis (Lyapunov, Robust, ISS)"
        echo "  ├── Regret Bounds (Information-theoretic, Sample complexity)"
        echo "  ├── Uncertainty Calibration (Bayesian, Predictive intervals)"
        echo "  ├── Safety Verification (Reachability, Barrier functions)"
        echo "  └── Appendix (Notation, Detailed proofs, Examples)"
    else
        print_error "PDF not found after compilation"
        exit 1
    fi
}

# Clean build files
clean_build() {
    print_status "Cleaning build files..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"/*
        print_success "Build directory cleaned"
    fi
}

# Open PDF if possible
open_pdf() {
    if [ -f "$BUILD_DIR/$MAIN_DOC.pdf" ]; then
        print_status "Attempting to open PDF..."
        
        if command_exists open; then
            # macOS
            open "$BUILD_DIR/$MAIN_DOC.pdf"
            print_success "PDF opened with default viewer (macOS)"
        elif command_exists xdg-open; then
            # Linux
            xdg-open "$BUILD_DIR/$MAIN_DOC.pdf"
            print_success "PDF opened with default viewer (Linux)"
        elif command_exists evince; then
            # Linux with Evince
            evince "$BUILD_DIR/$MAIN_DOC.pdf" &
            print_success "PDF opened with Evince"
        else
            print_warning "No PDF viewer found. Please open $BUILD_DIR/$MAIN_DOC.pdf manually"
        fi
    fi
}

# Show help
show_help() {
    echo "Mathematical Analysis Documentation Compilation Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --clean    Clean build directory before compilation"
    echo "  -v, --view     Open PDF after successful compilation"
    echo "  -q, --quick    Quick build (skip bibliography processing)"
    echo "  --check-deps   Check LaTeX dependencies only"
    echo "  --clean-only   Clean build directory and exit"
    echo
    echo "Examples:"
    echo "  $0              # Standard compilation"
    echo "  $0 -c -v        # Clean, compile, and view"
    echo "  $0 --quick      # Quick compilation without bibliography"
    echo "  $0 --check-deps # Check if LaTeX is properly installed"
}

# Parse command line arguments
CLEAN_BEFORE=false
VIEW_AFTER=false
QUICK_BUILD=false
CHECK_DEPS_ONLY=false
CLEAN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_BEFORE=true
            shift
            ;;
        -v|--view)
            VIEW_AFTER=true
            shift
            ;;
        -q|--quick)
            QUICK_BUILD=true
            shift
            ;;
        --check-deps)
            CHECK_DEPS_ONLY=true
            shift
            ;;
        --clean-only)
            CLEAN_ONLY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=========================================="
    echo "Mathematical Analysis Documentation Build"
    echo "=========================================="
    echo
    
    # Check dependencies first
    check_dependencies
    
    if [ "$CHECK_DEPS_ONLY" = true ]; then
        print_success "Dependency check completed successfully"
        exit 0
    fi
    
    # Clean if requested
    if [ "$CLEAN_BEFORE" = true ] || [ "$CLEAN_ONLY" = true ]; then
        clean_build
    fi
    
    if [ "$CLEAN_ONLY" = true ]; then
        exit 0
    fi
    
    # Setup build environment
    setup_build_dir
    
    # Compile document
    if [ "$QUICK_BUILD" = true ]; then
        print_status "Quick build mode - skipping bibliography processing"
        pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_DOC.tex" > "$BUILD_DIR/quick_build.log" 2>&1
        if [ $? -ne 0 ]; then
            print_error "Quick build failed. Check $BUILD_DIR/quick_build.log"
            exit 1
        fi
        print_success "Quick build completed"
    else
        compile_document
    fi
    
    # Generate statistics
    generate_stats
    
    # Open PDF if requested
    if [ "$VIEW_AFTER" = true ]; then
        open_pdf
    fi
    
    echo
    print_success "Mathematical analysis documentation build completed!"
    print_status "Output: $BUILD_DIR/$MAIN_DOC.pdf"
    echo
}

# Run main function
main