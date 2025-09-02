# Model-Based RL for Predictive Human Intent Recognition - Documentation Site

This repository contains the Jekyll documentation website for the Model-Based RL for Predictive Human Intent Recognition research project.

## Overview

This Jekyll site provides comprehensive documentation for our research on Bayesian Reinforcement Learning applied to human intent recognition in collaborative robotics. The site features:

- **Professional Documentation**: Clean, academic-style presentation
- **Interactive Features**: Table of contents, code copying, smooth navigation
- **Responsive Design**: Optimized for desktop, tablet, and mobile viewing
- **Accessibility**: WCAG-compliant design with proper semantic markup
- **Performance**: Optimized loading and rendering for fast access

## Quick Start

### Prerequisites

- Ruby 2.7+ (recommended: 3.1.0+)
- Bundler gem
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/model-based-rl-human-intent.git
cd model-based-rl-human-intent/jekyll_site

# Install dependencies
./build.sh install

# Build and serve locally
./build.sh serve
```

The site will be available at `http://localhost:4000`

## Build System

This project includes a comprehensive build script (`build.sh`) that automates common development tasks:

### Available Commands

```bash
# Development
./build.sh install     # Install dependencies
./build.sh build       # Build in development mode  
./build.sh serve       # Start development server
./build.sh serve 3000  # Start server on custom port

# Production
./build.sh build-prod  # Build for production
./build.sh optimize    # Optimize assets (requires Node.js tools)
./build.sh test        # Run validation and tests

# Utilities
./build.sh clean       # Clean build artifacts
./build.sh validate    # Validate configuration
./build.sh report      # Generate performance report
./build.sh full        # Complete build pipeline

# Deployment
./build.sh deploy github   # Deploy to GitHub Pages
./build.sh deploy netlify  # Build for Netlify
./build.sh deploy s3       # Deploy to AWS S3
```

### Full Build Pipeline

For production builds, run the complete pipeline:

```bash
./build.sh full
```

This will:
1. Clean previous build artifacts
2. Build the site for production
3. Optimize CSS, JavaScript, and images
4. Run validation tests
5. Generate a performance report

## Site Structure

```
jekyll_site/
├── _config.yml           # Jekyll configuration
├── _layouts/             # Page layouts
│   └── default.html      # Main layout template
├── assets/               # Static assets
│   ├── css/             # Stylesheets
│   │   ├── main.css     # Main styles
│   │   ├── syntax.css   # Code highlighting
│   │   └── print.css    # Print styles
│   └── js/              # JavaScript files
│       ├── main.js      # Main functionality
│       └── toc.js       # Table of contents generator
├── *.md                 # Content pages
│   ├── index.md         # Home page
│   ├── about.md         # About page
│   ├── methodology.md   # Technical details
│   ├── results.md       # Experimental results
│   ├── conclusion.md    # Conclusions and future work
│   └── contact.md       # Contact information
├── Gemfile              # Ruby dependencies
├── build.sh             # Build automation script
└── README.md            # This file
```

## Features

### Technical Features

- **Jekyll 4.3.2**: Latest stable Jekyll with GitHub Pages compatibility
- **Responsive Design**: Mobile-first CSS with breakpoints at 768px and 480px
- **Accessibility**: WCAG 2.1 AA compliance with semantic markup and ARIA labels
- **SEO Optimization**: Meta tags, Open Graph, Twitter Cards, and structured data
- **Performance**: Optimized CSS/JS, image compression, and gzip compression
- **Math Support**: MathJax integration for mathematical notation rendering

### Interactive Features

- **Mobile Navigation**: Hamburger menu with smooth animations
- **Smooth Scrolling**: Enhanced anchor link navigation
- **Code Copying**: One-click copying of code blocks
- **Table of Contents**: Auto-generated TOC with active section highlighting
- **Back to Top**: Floating button for long pages
- **Print Styles**: Optimized formatting for printing

### Content Features

- **Comprehensive Documentation**: Detailed coverage of methodology, results, and conclusions
- **Mathematical Notation**: Properly formatted equations using MathJax
- **Syntax Highlighting**: Code blocks with Rouge highlighter
- **Responsive Tables**: Mobile-optimized data presentation
- **Professional Typography**: Clean, readable fonts and spacing

## Configuration

### Site Settings (_config.yml)

Key configuration options:

```yaml
# Site information
title: "Model-Based RL for Predictive Human Intent Recognition"
description: "Research documentation for Bayesian RL human intent recognition"

# Build settings
markdown: kramdown
highlighter: rouge
plugins: [jekyll-feed, jekyll-sitemap, jekyll-seo-tag]

# Navigation menu
nav:
  - title: "Home"
    url: "/"
  - title: "About"
    url: "/about"
  # ... more navigation items
```

### Customization

#### Styling
- Edit `assets/css/main.css` for visual customization
- Modify `assets/css/syntax.css` for code highlighting themes
- Update `assets/css/print.css` for print formatting

#### Layout
- Customize `_layouts/default.html` for structural changes
- Add new layouts in the `_layouts/` directory

#### Functionality
- Extend `assets/js/main.js` for additional interactive features
- Modify `assets/js/toc.js` for table of contents behavior

## Development Workflow

### Local Development

1. **Start Development Server**:
   ```bash
   ./build.sh serve
   ```
   - Enables live reloading
   - Includes draft posts
   - Detailed error reporting

2. **Make Changes**:
   - Edit markdown files for content
   - Modify CSS/JS for styling and functionality
   - Update `_config.yml` for site settings

3. **Preview Changes**:
   - Site automatically rebuilds on file changes
   - Refresh browser to see updates
   - Check console for any errors

### Content Creation

#### Adding New Pages

1. Create a new `.md` file in the root directory
2. Add front matter with layout and metadata:
   ```yaml
   ---
   layout: default
   title: "Page Title"
   permalink: /page-url/
   ---
   ```
3. Add the page to navigation in `_config.yml`

#### Mathematical Content

Use MathJax syntax for mathematical expressions:

```markdown
Inline math: $E = mc^2$

Display math:
$$
\mathcal{L} = \mathbb{E}_{q(\theta)}[\log p(s_{t+1}|s_t, a_t, \theta)] - D_{KL}[q(\theta) || p(\theta)]
$$
```

#### Code Blocks

Use fenced code blocks with language specification:

```python
def bayesian_rl_update(replay_buffer, model_ensemble, policy):
    # Sample batch from replay buffer
    batch = replay_buffer.sample(batch_size)
    # ... implementation
```

### Testing and Validation

#### Automated Testing

```bash
# Run all tests
./build.sh test

# Individual validation steps
./build.sh validate    # Check configuration
bundle exec htmlproofer _site  # HTML validation
```

#### Manual Testing

- Test responsive design at different screen sizes
- Verify navigation and interactive features
- Check mathematical notation rendering
- Test print functionality
- Validate accessibility with screen readers

### Deployment

#### GitHub Pages

```bash
# Deploy to GitHub Pages
./build.sh deploy github
```

This automatically:
- Builds the site for production
- Commits to `gh-pages` branch
- Pushes to remote repository

#### Other Platforms

```bash
# Build for Netlify
./build.sh deploy netlify

# Deploy to AWS S3 (requires AWS CLI)
./build.sh deploy s3
```

## Performance Optimization

### Asset Optimization

The build system includes several optimization steps:

1. **CSS Minification**: Using csso-cli
2. **JavaScript Minification**: Using uglify-js
3. **Image Optimization**: Using imageoptim
4. **Gzip Compression**: Pre-compressed files for server delivery

### Performance Monitoring

Generate performance reports:

```bash
./build.sh report
```

This creates a detailed report including:
- File count and size analysis
- Largest files identification
- Build statistics

### Web Performance

The site is optimized for:
- **First Contentful Paint**: < 2s
- **Largest Contentful Paint**: < 4s
- **Cumulative Layout Shift**: < 0.1
- **First Input Delay**: < 100ms

## Browser Support

### Supported Browsers

- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile Browsers**: iOS Safari 14+, Chrome Mobile 90+
- **Legacy Support**: IE 11 (basic functionality only)

### Progressive Enhancement

The site uses progressive enhancement:
- Core content accessible without JavaScript
- Enhanced features available with JavaScript enabled
- Graceful degradation for unsupported features

## Accessibility

### WCAG 2.1 AA Compliance

- **Semantic HTML**: Proper heading hierarchy and landmark roles
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: ARIA labels and descriptions
- **Color Contrast**: Minimum 4.5:1 ratio for normal text
- **Focus Management**: Visible focus indicators

### Accessibility Features

- Skip navigation links
- Alt text for images
- Descriptive link text
- Proper form labeling
- High contrast mode support

## SEO Optimization

### Technical SEO

- **Meta Tags**: Title, description, and keywords
- **Open Graph**: Social media sharing optimization
- **Twitter Cards**: Enhanced Twitter sharing
- **Canonical URLs**: Prevent duplicate content
- **XML Sitemap**: Automatic generation via jekyll-sitemap

### Content SEO

- **Structured Data**: Schema.org markup for research content
- **Semantic HTML**: Proper use of headings and sections
- **Internal Linking**: Cross-references between pages
- **Image SEO**: Descriptive filenames and alt text

## Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clear cache and rebuild
./build.sh clean
./build.sh build
```

#### Dependency Issues

```bash
# Reinstall dependencies
rm -rf vendor/bundle
./build.sh install
```

#### Permission Issues

```bash
# Fix script permissions
chmod +x build.sh
```

### Error Messages

#### "Could not find gem"
- Run `./build.sh install` to install missing gems
- Check Ruby version compatibility

#### "Address already in use"
- Change port: `./build.sh serve 3001`
- Kill existing Jekyll processes

#### "Liquid Exception"
- Check for syntax errors in markdown files
- Validate YAML front matter

## Contributing

### Development Guidelines

1. **Code Style**: Follow existing conventions
2. **Testing**: Run full test suite before commits
3. **Documentation**: Update README for significant changes
4. **Accessibility**: Maintain WCAG compliance

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Run `./build.sh full` to validate
5. Submit pull request with description

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Support

### Documentation

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Guide](https://docs.github.com/pages)
- [Liquid Template Language](https://shopify.github.io/liquid/)

### Contact

- **Email**: research@anthropic.com
- **GitHub**: [anthropics/model-based-rl-human-intent](https://github.com/anthropics/model-based-rl-human-intent)
- **Issues**: Report bugs and feature requests on GitHub Issues

---

*This documentation site is part of the Model-Based RL for Predictive Human Intent Recognition research project by the Claude Code Research Team at Anthropic.*