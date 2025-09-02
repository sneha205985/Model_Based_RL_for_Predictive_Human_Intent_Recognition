#!/bin/bash

# ==========================================================================
# Jekyll Documentation Site Build Script
# Automates the build, test, and deployment process
# ==========================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SITE_DIR}/_site"
TEMP_DIR="${SITE_DIR}/.tmp"
RUBY_VERSION="3.1.0"
NODE_VERSION="18"

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_message $BLUE "========================================"
    print_message $BLUE "$1"
    print_message $BLUE "========================================"
}

print_success() {
    print_message $GREEN "✓ $1"
}

print_warning() {
    print_message $YELLOW "⚠ $1"
}

print_error() {
    print_message $RED "✗ $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Ruby version
check_ruby_version() {
    if command_exists ruby; then
        local current_version=$(ruby -v | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n1)
        print_success "Ruby version: $current_version"
        
        # Check if version is compatible
        if ! ruby -e "exit(RUBY_VERSION >= '2.7.0')" 2>/dev/null; then
            print_warning "Ruby version $current_version detected. Recommended: 3.1.0+"
        fi
    else
        print_error "Ruby not found. Please install Ruby $RUBY_VERSION or higher."
        exit 1
    fi
}

# Function to check and install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Check Ruby
    check_ruby_version
    
    # Install Bundler if not present
    if ! gem list bundler -i >/dev/null 2>&1; then
        print_message $YELLOW "Installing Bundler..."
        gem install bundler
    fi
    print_success "Bundler is installed"
    
    # Install Ruby gems
    print_message $YELLOW "Installing Ruby gems..."
    bundle config set --local path 'vendor/bundle'
    bundle install --quiet
    print_success "Ruby gems installed"
    
    # Check Node.js for optional tools
    if command_exists node; then
        local node_version=$(node -v | cut -d'v' -f2)
        print_success "Node.js version: $node_version"
        
        # Install npm packages if package.json exists
        if [ -f "package.json" ]; then
            print_message $YELLOW "Installing Node.js packages..."
            npm install --silent
            print_success "Node.js packages installed"
        fi
    else
        print_warning "Node.js not found. Some optimization features may not be available."
    fi
}

# Function to validate configuration
validate_config() {
    print_header "Validating Configuration"
    
    # Check if required files exist
    local required_files=("_config.yml" "Gemfile" "index.md")
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "$file found"
        else
            print_error "$file not found"
            exit 1
        fi
    done
    
    # Validate _config.yml syntax
    if bundle exec ruby -e "require 'yaml'; YAML.load_file('_config.yml')" >/dev/null 2>&1; then
        print_success "_config.yml syntax is valid"
    else
        print_error "_config.yml has syntax errors"
        exit 1
    fi
    
    # Check for required directories
    local required_dirs=("_layouts" "assets/css" "assets/js")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_success "Directory $dir exists"
        else
            print_warning "Directory $dir not found"
        fi
    done
}

# Function to clean build artifacts
clean_build() {
    print_header "Cleaning Build Artifacts"
    
    # Remove Jekyll build directory
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Removed $BUILD_DIR"
    fi
    
    # Remove temporary files
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        print_success "Removed $TEMP_DIR"
    fi
    
    # Remove Jekyll cache
    if [ -d ".jekyll-cache" ]; then
        rm -rf ".jekyll-cache"
        print_success "Removed Jekyll cache"
    fi
    
    # Remove Sass cache
    if [ -d ".sass-cache" ]; then
        rm -rf ".sass-cache"
        print_success "Removed Sass cache"
    fi
    
    print_success "Build artifacts cleaned"
}

# Function to build the site
build_site() {
    print_header "Building Jekyll Site"
    
    local build_env=${1:-"development"}
    export JEKYLL_ENV="$build_env"
    
    print_message $YELLOW "Building in $build_env mode..."
    
    # Build with appropriate settings
    if [ "$build_env" = "production" ]; then
        bundle exec jekyll build --config _config.yml --profile --trace
    else
        bundle exec jekyll build --config _config.yml --drafts --trace
    fi
    
    print_success "Jekyll build completed"
    
    # Generate build info
    cat > "$BUILD_DIR/build-info.json" <<EOF
{
    "build_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "build_environment": "$build_env",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo "unknown")",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")",
    "build_host": "$(hostname)",
    "ruby_version": "$(ruby -v)",
    "jekyll_version": "$(bundle exec jekyll -v)"
}
EOF
    print_success "Build information generated"
}

# Function to optimize assets
optimize_assets() {
    print_header "Optimizing Assets"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found. Run build first."
        return 1
    fi
    
    # Optimize CSS
    if command_exists csso; then
        print_message $YELLOW "Optimizing CSS files..."
        find "$BUILD_DIR" -name "*.css" -exec csso {} -o {} \;
        print_success "CSS files optimized"
    else
        print_warning "csso not found. Install with: npm install -g csso-cli"
    fi
    
    # Optimize JavaScript
    if command_exists uglifyjs; then
        print_message $YELLOW "Optimizing JavaScript files..."
        find "$BUILD_DIR" -name "*.js" -exec uglifyjs {} -o {} -m -c \;
        print_success "JavaScript files optimized"
    else
        print_warning "uglifyjs not found. Install with: npm install -g uglify-js"
    fi
    
    # Optimize images
    if command_exists imageoptim; then
        print_message $YELLOW "Optimizing images..."
        find "$BUILD_DIR" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | xargs imageoptim
        print_success "Images optimized"
    else
        print_warning "imageoptim not found. Image optimization skipped."
    fi
    
    # Generate gzipped versions for server compression
    print_message $YELLOW "Generating compressed files..."
    find "$BUILD_DIR" -name "*.html" -o -name "*.css" -o -name "*.js" -o -name "*.xml" | while read file; do
        gzip -c "$file" > "$file.gz"
    done
    print_success "Compressed files generated"
}

# Function to run tests
run_tests() {
    print_header "Running Tests"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found. Run build first."
        return 1
    fi
    
    # HTML validation with htmlproofer
    if bundle list | grep -q htmlproofer; then
        print_message $YELLOW "Running HTML validation..."
        
        # Run htmlproofer with appropriate options
        bundle exec htmlproofer "$BUILD_DIR" \
            --check-html \
            --check-img-http \
            --check-opengraph \
            --report-missing-names \
            --log-level :warn \
            --assume-extension \
            --url-ignore "/linkedin.com/,/twitter.com/" \
            --file-ignore "/vendor/"
        
        print_success "HTML validation passed"
    else
        print_warning "htmlproofer not installed. Add to Gemfile for validation."
    fi
    
    # Check for broken internal links
    print_message $YELLOW "Checking internal links..."
    if find "$BUILD_DIR" -name "*.html" -exec grep -l "href=\"#" {} \; | wc -l | grep -q '^0$'; then
        print_success "No broken internal links found"
    else
        print_warning "Found potential internal link issues"
    fi
    
    # Validate JSON files
    print_message $YELLOW "Validating JSON files..."
    find "$BUILD_DIR" -name "*.json" | while read json_file; do
        if python3 -m json.tool "$json_file" >/dev/null 2>&1; then
            print_success "$(basename "$json_file") is valid JSON"
        else
            print_error "$(basename "$json_file") has JSON syntax errors"
        fi
    done
    
    # Check for accessibility issues
    if command_exists axe; then
        print_message $YELLOW "Running accessibility checks..."
        # This would require axe-cli to be installed
        print_warning "Accessibility checks require manual setup"
    fi
    
    print_success "All tests completed"
}

# Function to generate performance report
performance_report() {
    print_header "Generating Performance Report"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found. Run build first."
        return 1
    fi
    
    local report_file="$BUILD_DIR/performance-report.txt"
    
    echo "Performance Report - Generated $(date)" > "$report_file"
    echo "==========================================" >> "$report_file"
    echo >> "$report_file"
    
    # Site statistics
    echo "Site Statistics:" >> "$report_file"
    echo "- Total files: $(find "$BUILD_DIR" -type f | wc -l)" >> "$report_file"
    echo "- HTML files: $(find "$BUILD_DIR" -name "*.html" | wc -l)" >> "$report_file"
    echo "- CSS files: $(find "$BUILD_DIR" -name "*.css" | wc -l)" >> "$report_file"
    echo "- JS files: $(find "$BUILD_DIR" -name "*.js" | wc -l)" >> "$report_file"
    echo "- Image files: $(find "$BUILD_DIR" \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.gif" -o -name "*.svg" \) | wc -l)" >> "$report_file"
    echo >> "$report_file"
    
    # Size analysis
    echo "Size Analysis:" >> "$report_file"
    echo "- Total site size: $(du -sh "$BUILD_DIR" | cut -f1)" >> "$report_file"
    echo "- HTML size: $(find "$BUILD_DIR" -name "*.html" -exec cat {} \; | wc -c | numfmt --to=iec)" >> "$report_file"
    echo "- CSS size: $(find "$BUILD_DIR" -name "*.css" -exec cat {} \; | wc -c | numfmt --to=iec)" >> "$report_file"
    echo "- JS size: $(find "$BUILD_DIR" -name "*.js" -exec cat {} \; | wc -c | numfmt --to=iec)" >> "$report_file"
    echo >> "$report_file"
    
    # Largest files
    echo "Largest Files:" >> "$report_file"
    find "$BUILD_DIR" -type f -exec du -h {} \; | sort -hr | head -10 >> "$report_file"
    echo >> "$report_file"
    
    print_success "Performance report generated: $report_file"
}

# Function to serve site locally
serve_site() {
    print_header "Starting Development Server"
    
    local port=${1:-4000}
    
    print_message $YELLOW "Starting Jekyll server on port $port..."
    print_message $BLUE "Site will be available at: http://localhost:$port"
    print_message $BLUE "Press Ctrl+C to stop the server"
    echo
    
    bundle exec jekyll serve --port "$port" --livereload --drafts --trace
}

# Function to deploy site
deploy_site() {
    print_header "Deploying Site"
    
    local target=${1:-"github"}
    
    case "$target" in
        "github")
            print_message $YELLOW "Deploying to GitHub Pages..."
            
            # Check if gh-pages branch exists
            if git show-ref --verify --quiet refs/heads/gh-pages; then
                git checkout gh-pages
                git reset --hard main
            else
                git checkout -b gh-pages
            fi
            
            # Build for production
            build_site "production"
            
            # Commit and push
            git add .
            git commit -m "Deploy site - $(date)"
            git push origin gh-pages
            git checkout main
            
            print_success "Deployed to GitHub Pages"
            ;;
        "netlify")
            print_message $YELLOW "Building for Netlify deployment..."
            build_site "production"
            print_success "Build ready for Netlify deployment"
            print_message $BLUE "Upload the _site directory to Netlify"
            ;;
        "s3")
            print_message $YELLOW "Deploying to AWS S3..."
            if command_exists aws; then
                build_site "production"
                aws s3 sync "$BUILD_DIR" s3://your-bucket-name --delete
                print_success "Deployed to S3"
            else
                print_error "AWS CLI not found. Install aws-cli first."
            fi
            ;;
        *)
            print_error "Unknown deployment target: $target"
            print_message $BLUE "Supported targets: github, netlify, s3"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
Jekyll Documentation Site Build Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install     Install dependencies
    validate    Validate configuration
    clean       Clean build artifacts
    build       Build the site (default: development)
    build-prod  Build for production
    optimize    Optimize built assets
    test        Run tests and validation
    report      Generate performance report
    serve       Start development server
    deploy      Deploy site
    full        Full build pipeline (clean, build, optimize, test)
    help        Show this help message

Examples:
    $0 install              # Install dependencies
    $0 build                # Build in development mode
    $0 build-prod           # Build for production
    $0 serve                # Start development server
    $0 serve 3000           # Start server on port 3000
    $0 deploy github        # Deploy to GitHub Pages
    $0 full                 # Run complete build pipeline

For more information, see the documentation at:
https://jekyllrb.com/docs/
EOF
}

# Main script logic
main() {
    cd "$SITE_DIR"
    
    case "${1:-help}" in
        "install")
            install_dependencies
            ;;
        "validate")
            validate_config
            ;;
        "clean")
            clean_build
            ;;
        "build")
            build_site "development"
            ;;
        "build-prod")
            build_site "production"
            ;;
        "optimize")
            optimize_assets
            ;;
        "test")
            run_tests
            ;;
        "report")
            performance_report
            ;;
        "serve")
            serve_site "${2:-4000}"
            ;;
        "deploy")
            deploy_site "${2:-github}"
            ;;
        "full")
            clean_build
            build_site "production"
            optimize_assets
            run_tests
            performance_report
            print_message $GREEN "Full build pipeline completed successfully!"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"