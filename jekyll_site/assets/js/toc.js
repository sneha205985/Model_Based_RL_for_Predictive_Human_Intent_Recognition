/**
 * Table of Contents Generator
 * Automatically generates a table of contents from page headings
 */

(function() {
    'use strict';

    function generateTableOfContents() {
        const tocContainer = document.getElementById('toc');
        const contentArea = document.querySelector('.page-content');
        
        if (!tocContainer || !contentArea) {
            return;
        }

        // Find all headings in the content area
        const headings = contentArea.querySelectorAll('h1, h2, h3, h4, h5, h6');
        
        if (headings.length === 0) {
            tocContainer.innerHTML = '<p>No headings found.</p>';
            return;
        }

        // Generate IDs for headings that don't have them
        headings.forEach((heading, index) => {
            if (!heading.id) {
                const text = heading.textContent.trim();
                const id = text.toLowerCase()
                    .replace(/[^\w\s-]/g, '') // Remove special characters
                    .replace(/\s+/g, '-')     // Replace spaces with hyphens
                    .replace(/--+/g, '-')     // Replace multiple hyphens with single
                    .replace(/^-+|-+$/g, ''); // Remove leading/trailing hyphens
                
                heading.id = id || `heading-${index}`;
            }
        });

        // Build TOC structure
        const tocList = buildTocStructure(headings);
        tocContainer.appendChild(tocList);

        // Add smooth scrolling and active highlighting
        addTocInteractivity();
        
        // Update active section on scroll
        updateActiveSection();
        window.addEventListener('scroll', throttle(updateActiveSection, 100));
    }

    function buildTocStructure(headings) {
        const tocFragment = document.createDocumentFragment();
        const stack = []; // Stack to manage nested lists
        let currentList = document.createElement('ul');
        currentList.className = 'toc-list';
        tocFragment.appendChild(currentList);

        headings.forEach(heading => {
            const level = parseInt(heading.tagName.charAt(1));
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent.trim();
            link.className = 'toc-link';
            link.setAttribute('data-level', level);
            
            listItem.appendChild(link);

            // Handle nesting based on heading level
            if (stack.length === 0 || level <= stack[stack.length - 1].level) {
                // Same level or higher level heading
                while (stack.length > 0 && level <= stack[stack.length - 1].level) {
                    stack.pop();
                }
                
                if (stack.length === 0) {
                    currentList.appendChild(listItem);
                } else {
                    stack[stack.length - 1].list.appendChild(listItem);
                }
            } else {
                // Lower level heading (nested)
                const nestedList = document.createElement('ul');
                nestedList.className = 'toc-nested';
                
                if (stack.length > 0) {
                    const parentItem = stack[stack.length - 1].item;
                    parentItem.appendChild(nestedList);
                } else {
                    currentList.appendChild(nestedList);
                }
                
                nestedList.appendChild(listItem);
                currentList = nestedList;
            }

            // Add to stack
            stack.push({
                level: level,
                item: listItem,
                list: currentList
            });
        });

        return tocFragment;
    }

    function addTocInteractivity() {
        const tocLinks = document.querySelectorAll('.toc-link');
        
        tocLinks.forEach(link => {
            link.addEventListener('click', function(event) {
                event.preventDefault();
                
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    // Remove active class from all links
                    tocLinks.forEach(l => l.classList.remove('active'));
                    
                    // Add active class to clicked link
                    this.classList.add('active');
                    
                    // Smooth scroll to target
                    const headerOffset = 80; // Account for sticky header
                    const elementPosition = targetElement.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    function updateActiveSection() {
        const tocLinks = document.querySelectorAll('.toc-link');
        const headings = document.querySelectorAll('.page-content h1, .page-content h2, .page-content h3, .page-content h4, .page-content h5, .page-content h6');
        
        if (headings.length === 0) return;

        let activeHeading = null;
        const scrollPosition = window.pageYOffset + 100; // Offset for better UX

        // Find the current active heading
        headings.forEach(heading => {
            const headingTop = heading.getBoundingClientRect().top + window.pageYOffset;
            
            if (headingTop <= scrollPosition) {
                activeHeading = heading;
            }
        });

        // Update TOC active states
        tocLinks.forEach(link => {
            link.classList.remove('active');
            
            if (activeHeading && link.getAttribute('href') === `#${activeHeading.id}`) {
                link.classList.add('active');
                
                // Scroll TOC if needed to keep active link visible
                scrollTocToActiveLink(link);
            }
        });
    }

    function scrollTocToActiveLink(activeLink) {
        const tocContainer = document.querySelector('.table-of-contents');
        if (!tocContainer) return;

        const containerRect = tocContainer.getBoundingClientRect();
        const linkRect = activeLink.getBoundingClientRect();

        // Check if the active link is outside the visible area
        if (linkRect.top < containerRect.top || linkRect.bottom > containerRect.bottom) {
            // Calculate scroll position to center the active link
            const scrollTop = activeLink.offsetTop - (tocContainer.offsetHeight / 2);
            
            tocContainer.scrollTo({
                top: Math.max(0, scrollTop),
                behavior: 'smooth'
            });
        }
    }

    // Throttle function to limit scroll event frequency
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Collapse/Expand TOC on mobile
    function initTocCollapse() {
        const tocContainer = document.querySelector('.table-of-contents');
        if (!tocContainer) return;

        const tocHeader = tocContainer.querySelector('h2');
        if (!tocHeader) return;

        // Add collapse functionality on smaller screens
        function handleTocCollapse() {
            if (window.innerWidth <= 768) {
                tocHeader.style.cursor = 'pointer';
                tocHeader.setAttribute('aria-expanded', 'true');
                tocHeader.setAttribute('role', 'button');
                tocHeader.setAttribute('tabindex', '0');
                
                const tocContent = tocContainer.querySelector('.toc-list');
                
                tocHeader.addEventListener('click', toggleToc);
                tocHeader.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        toggleToc();
                    }
                });
                
                function toggleToc() {
                    const isExpanded = tocHeader.getAttribute('aria-expanded') === 'true';
                    tocHeader.setAttribute('aria-expanded', !isExpanded);
                    tocContent.style.display = isExpanded ? 'none' : 'block';
                    
                    // Update header text or add icon
                    const icon = isExpanded ? ' ▼' : ' ▲';
                    tocHeader.textContent = tocHeader.textContent.replace(/ [▼▲]$/, '') + icon;
                }
            }
        }

        // Initialize and handle window resize
        handleTocCollapse();
        window.addEventListener('resize', throttle(handleTocCollapse, 250));
    }

    // Enhanced TOC styling
    function addTocStyling() {
        const style = document.createElement('style');
        style.textContent = `
            .toc-list {
                list-style: none;
                margin: 0;
                padding: 0;
            }
            
            .toc-nested {
                list-style: none;
                margin: 0.25rem 0 0 1rem;
                padding: 0;
                border-left: 1px solid #e1e4e8;
            }
            
            .toc-link {
                display: block;
                padding: 0.375rem 0.5rem;
                color: #555;
                text-decoration: none;
                font-size: 0.875rem;
                line-height: 1.4;
                border-radius: 3px;
                transition: all 0.2s ease;
            }
            
            .toc-link:hover {
                color: #0066cc;
                background-color: #f6f8fa;
                text-decoration: none;
            }
            
            .toc-link.active {
                color: #0066cc;
                background-color: #e3f2fd;
                font-weight: 500;
                border-left: 3px solid #0066cc;
                padding-left: calc(0.5rem - 3px);
            }
            
            .toc-link[data-level="1"] {
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .toc-link[data-level="2"] {
                font-size: 0.875rem;
            }
            
            .toc-link[data-level="3"],
            .toc-link[data-level="4"],
            .toc-link[data-level="5"],
            .toc-link[data-level="6"] {
                font-size: 0.8125rem;
                color: #666;
            }
            
            @media (max-width: 768px) {
                .table-of-contents {
                    margin: 1rem 0;
                }
                
                .table-of-contents h2 {
                    user-select: none;
                }
                
                .table-of-contents h2::after {
                    content: " ▲";
                    float: right;
                    font-size: 0.8rem;
                    color: #666;
                }
            }
        `;
        
        document.head.appendChild(style);
    }

    // Initialize TOC when DOM is ready
    function initToc() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                addTocStyling();
                generateTableOfContents();
                initTocCollapse();
            });
        } else {
            addTocStyling();
            generateTableOfContents();
            initTocCollapse();
        }
    }

    // Export for potential external use
    window.TocGenerator = {
        generateTableOfContents,
        updateActiveSection,
        initToc
    };

    // Auto-initialize
    initToc();

})();