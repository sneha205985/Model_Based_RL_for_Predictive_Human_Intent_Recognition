# Makefile for LaTeX mathematical analysis documentation

# Document name
MAIN = main_document
BIB = references

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Directories
SRCDIR = .
BUILDDIR = build
FIGDIR = figures

# Create build directory if it doesn't exist
$(shell mkdir -p $(BUILDDIR))

# Default target
all: $(BUILDDIR)/$(MAIN).pdf

# Main PDF target
$(BUILDDIR)/$(MAIN).pdf: $(MAIN).tex $(BIB).bib $(wildcard */*.tex) $(wildcard appendix/*.tex)
	@echo "Building mathematical analysis documentation..."
	$(LATEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(BIBTEX) $(BUILDDIR)/$(MAIN)
	$(LATEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	@echo "Documentation built successfully: $(BUILDDIR)/$(MAIN).pdf"

# Clean build files
clean:
	rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned"

# Clean everything including PDF
distclean: clean
	rm -f $(BUILDDIR)/$(MAIN).pdf
	@echo "All build files removed"

# View PDF (requires PDF viewer)
view: $(BUILDDIR)/$(MAIN).pdf
	@if command -v open >/dev/null 2>&1; then \
		open $(BUILDDIR)/$(MAIN).pdf; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(BUILDDIR)/$(MAIN).pdf; \
	elif command -v evince >/dev/null 2>&1; then \
		evince $(BUILDDIR)/$(MAIN).pdf; \
	else \
		echo "No PDF viewer found. Please open $(BUILDDIR)/$(MAIN).pdf manually."; \
	fi

# Quick build (no bibliography)
quick: $(MAIN).tex
	$(LATEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	@echo "Quick build completed"

# Check for required LaTeX packages
check-deps:
	@echo "Checking LaTeX dependencies..."
	@latex --version > /dev/null 2>&1 || (echo "LaTeX not found. Please install LaTeX distribution." && exit 1)
	@bibtex --version > /dev/null 2>&1 || (echo "BibTeX not found. Please install BibTeX." && exit 1)
	@echo "All dependencies satisfied"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the complete mathematical analysis document"
	@echo "  quick      - Quick build without bibliography"
	@echo "  clean      - Remove build files (keep PDF)"
	@echo "  distclean  - Remove all build files including PDF"
	@echo "  view       - Open the PDF in default viewer"
	@echo "  check-deps - Check for required LaTeX dependencies"
	@echo "  help       - Show this help message"

# Phony targets
.PHONY: all clean distclean view quick check-deps help

# Build statistics
stats: $(BUILDDIR)/$(MAIN).pdf
	@echo "Document statistics:"
	@echo "  PDF size: $$(du -h $(BUILDDIR)/$(MAIN).pdf | cut -f1)"
	@echo "  Pages: $$(pdfinfo $(BUILDDIR)/$(MAIN).pdf 2>/dev/null | grep Pages | awk '{print $$2}' || echo 'N/A')"
	@echo "  Words (approx): $$(cat $(wildcard */*.tex) appendix/*.tex $(MAIN).tex 2>/dev/null | wc -w || echo 'N/A')"