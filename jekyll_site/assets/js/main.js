/**
 * Model-Based RL Documentation Site - Main JavaScript
 * Handles navigation, interactive features, and user experience enhancements
 */

(function() {
    'use strict';

    // DOM Ready Handler
    function domReady(callback) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', callback);
        } else {
            callback();
        }
    }

    // Mobile Navigation Toggle
    function initMobileNavigation() {
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');

        if (!navToggle || !navMenu) return;

        navToggle.addEventListener('click', function() {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            
            // Toggle ARIA state
            navToggle.setAttribute('aria-expanded', !isExpanded);
            
            // Toggle classes
            navToggle.classList.toggle('active');
            navMenu.classList.toggle('active');
            
            // Prevent body scroll when menu is open
            document.body.classList.toggle('nav-open', !isExpanded);
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!navToggle.contains(event.target) && !navMenu.contains(event.target)) {
                navToggle.setAttribute('aria-expanded', 'false');
                navToggle.classList.remove('active');
                navMenu.classList.remove('active');
                document.body.classList.remove('nav-open');
            }
        });

        // Close menu on escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && navMenu.classList.contains('active')) {
                navToggle.setAttribute('aria-expanded', 'false');
                navToggle.classList.remove('active');
                navMenu.classList.remove('active');
                document.body.classList.remove('nav-open');
                navToggle.focus(); // Return focus to toggle button
            }
        });
    }

    // Smooth Scrolling for Anchor Links
    function initSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        
        anchorLinks.forEach(link => {
            link.addEventListener('click', function(event) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    event.preventDefault();
                    
                    // Close mobile menu if open
                    const navMenu = document.querySelector('.nav-menu');
                    const navToggle = document.querySelector('.nav-toggle');
                    if (navMenu && navMenu.classList.contains('active')) {
                        navToggle.setAttribute('aria-expanded', 'false');
                        navToggle.classList.remove('active');
                        navMenu.classList.remove('active');
                        document.body.classList.remove('nav-open');
                    }
                    
                    // Smooth scroll to target
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update URL without triggering scroll
                    history.pushState(null, null, targetId);
                }
            });
        });
    }

    // Code Block Copy Functionality
    function initCodeCopyButtons() {
        const codeBlocks = document.querySelectorAll('.highlight');
        
        codeBlocks.forEach((block, index) => {
            // Create copy button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-btn';
            copyButton.textContent = 'Copy';
            copyButton.setAttribute('aria-label', 'Copy code to clipboard');
            copyButton.setAttribute('data-code-block', index);
            
            // Add button to code block
            block.style.position = 'relative';
            block.appendChild(copyButton);
            
            // Handle copy functionality
            copyButton.addEventListener('click', async function() {
                const codeElement = block.querySelector('code');
                if (!codeElement) return;
                
                try {
                    await navigator.clipboard.writeText(codeElement.textContent);
                    
                    // Visual feedback
                    copyButton.textContent = 'Copied!';
                    copyButton.classList.add('copied');
                    
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                        copyButton.classList.remove('copied');
                    }, 2000);
                    
                } catch (err) {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = codeElement.textContent;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    copyButton.textContent = 'Copied!';
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                    }, 2000);
                }
            });
        });
    }

    // Back to Top Button
    function initBackToTop() {
        // Create back to top button
        const backToTopButton = document.createElement('button');
        backToTopButton.className = 'back-to-top';
        backToTopButton.innerHTML = '↑';
        backToTopButton.setAttribute('aria-label', 'Back to top');
        backToTopButton.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 50%;
            width: 3rem;
            height: 3rem;
            font-size: 1.2rem;
            cursor: pointer;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        document.body.appendChild(backToTopButton);
        
        // Show/hide based on scroll position
        function toggleBackToTop() {
            if (window.pageYOffset > 300) {
                backToTopButton.style.opacity = '1';
                backToTopButton.style.visibility = 'visible';
            } else {
                backToTopButton.style.opacity = '0';
                backToTopButton.style.visibility = 'hidden';
            }
        }
        
        // Throttled scroll handler
        let ticking = false;
        function handleScroll() {
            if (!ticking) {
                requestAnimationFrame(function() {
                    toggleBackToTop();
                    ticking = false;
                });
                ticking = true;
            }
        }
        
        window.addEventListener('scroll', handleScroll);
        
        // Click handler
        backToTopButton.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Enhanced Focus Management
    function initAccessibility() {
        // Skip link functionality
        const skipLink = document.querySelector('.skip-link');
        if (skipLink) {
            skipLink.addEventListener('click', function(event) {
                event.preventDefault();
                const target = document.querySelector('#main-content');
                if (target) {
                    target.focus();
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        }

        // Enhance keyboard navigation
        document.addEventListener('keydown', function(event) {
            // Escape key handling for modals and overlays
            if (event.key === 'Escape') {
                // Close any open dropdowns or modals
                const activeElements = document.querySelectorAll('.active');
                activeElements.forEach(element => {
                    if (element.classList.contains('nav-menu')) {
                        const navToggle = document.querySelector('.nav-toggle');
                        if (navToggle) {
                            navToggle.setAttribute('aria-expanded', 'false');
                            navToggle.classList.remove('active');
                            element.classList.remove('active');
                            document.body.classList.remove('nav-open');
                        }
                    }
                });
            }
        });
    }

    // Math Rendering Enhancement (for MathJax)
    function enhanceMathRendering() {
        // Add loading indicators for math content
        const mathElements = document.querySelectorAll('.MathJax, .MathJax_Display');
        mathElements.forEach(element => {
            element.style.minHeight = '1em';
        });

        // Handle MathJax rendering completion
        if (window.MathJax) {
            MathJax.Hub.Register.MessageHook('End Process', function() {
                // Remove loading indicators and enhance accessibility
                const processedMath = document.querySelectorAll('[id^="MathJax-Element-"]');
                processedMath.forEach(element => {
                    element.setAttribute('role', 'math');
                    element.setAttribute('aria-label', 'Mathematical expression');
                });
            });
        }
    }

    // Performance Monitoring
    function initPerformanceMonitoring() {
        // Monitor Core Web Vitals if supported
        if ('web-vital' in window) {
            // This would integrate with actual performance monitoring
            // For now, we'll just log page load performance
            window.addEventListener('load', function() {
                const navigationTiming = performance.getEntriesByType('navigation')[0];
                if (navigationTiming) {
                    console.log('Page load performance:', {
                        domContentLoaded: navigationTiming.domContentLoadedEventEnd - navigationTiming.domContentLoadedEventStart,
                        loadComplete: navigationTiming.loadEventEnd - navigationTiming.loadEventStart,
                        totalTime: navigationTiming.loadEventEnd - navigationTiming.fetchStart
                    });
                }
            });
        }
    }

    // Search Enhancement (if search is implemented)
    function initSearchEnhancement() {
        const searchInput = document.querySelector('input[type="search"]');
        if (searchInput) {
            let searchTimeout;
            
            searchInput.addEventListener('input', function() {
                clearTimeout(searchTimeout);
                const query = this.value.trim();
                
                if (query.length >= 3) {
                    searchTimeout = setTimeout(() => {
                        performSearch(query);
                    }, 300);
                }
            });
        }
    }

    function performSearch(query) {
        // Placeholder for search functionality
        // This would integrate with a search service or static search
        console.log('Searching for:', query);
    }

    // Lazy Loading for Images
    function initLazyLoading() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.removeAttribute('data-src');
                            imageObserver.unobserve(img);
                        }
                    }
                });
            });

            // Observe images with data-src attribute
            const lazyImages = document.querySelectorAll('img[data-src]');
            lazyImages.forEach(img => imageObserver.observe(img));
        }
    }

    // Print Styles Enhancement
    function initPrintEnhancement() {
        window.addEventListener('beforeprint', function() {
            // Expand all collapsed sections for printing
            const collapsedElements = document.querySelectorAll('.collapsed');
            collapsedElements.forEach(element => {
                element.classList.add('print-expanded');
            });
        });

        window.addEventListener('afterprint', function() {
            // Restore collapsed state after printing
            const expandedElements = document.querySelectorAll('.print-expanded');
            expandedElements.forEach(element => {
                element.classList.remove('print-expanded');
            });
        });
    }

    // Error Handling
    function initErrorHandling() {
        window.addEventListener('error', function(event) {
            console.error('JavaScript error:', event.error);
            // Could implement error reporting here
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', function(event) {
            console.error('Unhandled promise rejection:', event.reason);
            // Could implement error reporting here
        });
    }

    // Initialize all functionality when DOM is ready
    domReady(function() {
        console.log('Initializing Model-Based RL Documentation Site');
        
        // Initialize all components
        initMobileNavigation();
        initSmoothScrolling();
        initCodeCopyButtons();
        initBackToTop();
        initAccessibility();
        enhanceMathRendering();
        initPerformanceMonitoring();
        initSearchEnhancement();
        initLazyLoading();
        initPrintEnhancement();
        initErrorHandling();
        
        console.log('Site initialization complete');
    });

    // Export functions for potential external use
    window.SiteJS = {
        initMobileNavigation,
        initSmoothScrolling,
        initCodeCopyButtons,
        initBackToTop,
        initAccessibility
    };

})();