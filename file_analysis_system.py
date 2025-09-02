#!/usr/bin/env python3
"""
Comprehensive File Analysis and Backup System
Phase 1: ANALYSIS ONLY - NO FILE DELETION

This script analyzes the entire project structure and classifies files for safe cleanup preparation.
"""

import os
import ast
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import time
from datetime import datetime

class FileAnalysisSystem:
    """Comprehensive file analysis for safe cleanup preparation"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_results = {
            'CRITICAL': [],
            'WORKING': [],
            'DUPLICATE': [],
            'GENERATED': [],
            'OBSOLETE': [],
            'UNKNOWN': []
        }
        
        # File patterns
        self.generated_patterns = {
            '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.DS_Store',
            '*.egg-info', 'build/', 'dist/', '.pytest_cache',
            '.coverage', 'htmlcov/', '.tox/', '.mypy_cache'
        }
        
        self.test_patterns = {
            'test_*.py', '*_test.py', 'tests/', 'conftest.py'
        }
        
        # Import tracking
        self.import_graph = defaultdict(set)
        self.imported_modules = set()
        self.all_python_files = set()
        
        # Git information
        self.git_info = {}
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete file analysis pipeline"""
        print("🔍 Starting Comprehensive File Analysis...")
        print(f"📁 Project Root: {self.project_root}")
        
        start_time = time.time()
        
        # Step 1: Discover all files
        print("\n1️⃣  Scanning directory structure...")
        all_files = self._scan_directory_structure()
        
        # Step 2: Analyze Git information
        print("\n2️⃣  Analyzing Git history...")
        self._analyze_git_history()
        
        # Step 3: Build dependency graph
        print("\n3️⃣  Building dependency graph...")
        self._build_dependency_graph()
        
        # Step 4: Classify files
        print("\n4️⃣  Classifying files...")
        self._classify_all_files(all_files)
        
        # Step 5: Generate analysis report
        print("\n5️⃣  Generating analysis report...")
        analysis_time = time.time() - start_time
        
        report = self._generate_analysis_report(analysis_time)
        
        # Step 6: Save results
        print("\n6️⃣  Saving analysis results...")
        self._save_analysis_results(report)
        
        print(f"\n✅ Analysis Complete! ({analysis_time:.2f}s)")
        return report
    
    def _scan_directory_structure(self) -> List[Path]:
        """Scan entire directory structure"""
        all_files = []
        total_size = 0
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') or d in {'.git'}]
            
            for file in files:
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    all_files.append(file_path)
                except (OSError, PermissionError):
                    continue
        
        print(f"   📊 Found {len(all_files)} files ({total_size / 1024 / 1024:.1f} MB)")
        return all_files
    
    def _analyze_git_history(self):
        """Analyze Git commit history for file activity"""
        try:
            # Get recent commits for each file
            result = subprocess.run(
                ['git', 'log', '--name-only', '--pretty=format:', '--since=30 days ago'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                recent_files = set()
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('commit'):
                        recent_files.add(line)
                
                self.git_info['recent_files'] = recent_files
                print(f"   📈 {len(recent_files)} files modified in last 30 days")
            
            # Get file creation dates
            result = subprocess.run(
                ['git', 'ls-files', '-z'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                git_files = set(result.stdout.split('\0'))
                self.git_info['tracked_files'] = git_files
                print(f"   📋 {len(git_files)} files tracked by Git")
                
        except subprocess.SubprocessError as e:
            print(f"   ⚠️  Git analysis failed: {e}")
            self.git_info = {'recent_files': set(), 'tracked_files': set()}
    
    def _build_dependency_graph(self):
        """Build import dependency graph"""
        python_files = list(self.project_root.rglob('*.py'))
        self.all_python_files = set(python_files)
        
        print(f"   🐍 Analyzing {len(python_files)} Python files...")
        
        for py_file in python_files:
            try:
                imports = self._extract_imports(py_file)
                rel_path = py_file.relative_to(self.project_root)
                self.import_graph[str(rel_path)] = imports
                self.imported_modules.update(imports)
            except Exception as e:
                print(f"   ⚠️  Could not analyze {py_file}: {e}")
        
        print(f"   🔗 Found {len(self.imported_modules)} unique imports")
    
    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract import statements from Python file"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        
        except (SyntaxError, UnicodeDecodeError, OSError):
            # Fallback to regex for problematic files
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple regex patterns
                import_patterns = [
                    r'^\s*import\s+([^\s,]+)',
                    r'^\s*from\s+([^\s]+)\s+import',
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    imports.update(matches)
            
            except OSError:
                pass
        
        return imports
    
    def _classify_all_files(self, all_files: List[Path]):
        """Classify all files into categories"""
        
        for file_path in all_files:
            try:
                rel_path = file_path.relative_to(self.project_root)
                classification = self._classify_single_file(file_path, rel_path)
                self.analysis_results[classification].append({
                    'path': str(rel_path),
                    'absolute_path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime,
                    'classification_reason': self._get_classification_reason(file_path, rel_path, classification)
                })
            except (OSError, ValueError):
                continue
        
        print(f"   📋 Classification complete:")
        for category, files in self.analysis_results.items():
            print(f"      {category}: {len(files)} files")
    
    def _classify_single_file(self, file_path: Path, rel_path: Path) -> str:
        """Classify a single file"""
        file_str = str(rel_path)
        
        # Check if generated file
        if self._is_generated_file(file_path, rel_path):
            return 'GENERATED'
        
        # Check if Python file for more detailed analysis
        if file_path.suffix == '.py':
            return self._classify_python_file(file_path, rel_path)
        
        # Check if critical non-Python file
        if self._is_critical_file(file_path, rel_path):
            return 'CRITICAL'
        
        # Check if working file (recently modified)
        if self._is_working_file(file_path, rel_path):
            return 'WORKING'
        
        # Check for duplicates
        if self._is_duplicate_file(file_path, rel_path):
            return 'DUPLICATE'
        
        # Default to unknown for further review
        return 'UNKNOWN'
    
    def _classify_python_file(self, file_path: Path, rel_path: Path) -> str:
        """Detailed classification for Python files"""
        file_str = str(rel_path)
        
        # Check if this module is imported by others
        module_name = self._path_to_module_name(rel_path)
        is_imported = any(module_name in imports for imports in self.import_graph.values())
        
        # Check if it imports critical modules
        imports_critical = bool(self.import_graph.get(file_str, set()))
        
        # Check if it's a test file
        is_test = any(pattern.replace('*', '') in file_str for pattern in self.test_patterns)
        
        # Check if recently active
        is_recent = file_str in self.git_info.get('recent_files', set())
        
        # Classification logic
        if is_imported and imports_critical:
            return 'CRITICAL'
        elif is_imported or (imports_critical and is_recent):
            return 'WORKING'
        elif is_test and is_recent:
            return 'WORKING'
        elif is_test and not is_recent:
            return 'OBSOLETE'
        elif is_recent:
            return 'WORKING'
        elif not imports_critical and not is_imported:
            return 'OBSOLETE'
        else:
            return 'UNKNOWN'
    
    def _is_generated_file(self, file_path: Path, rel_path: Path) -> bool:
        """Check if file is generated"""
        file_str = str(rel_path)
        
        # Check patterns
        for pattern in self.generated_patterns:
            if pattern.endswith('/'):
                if pattern[:-1] in file_str:
                    return True
            else:
                pattern_clean = pattern.replace('*', '')
                if pattern_clean in file_str:
                    return True
        
        # Check specific file content indicators
        if file_path.suffix == '.py':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_lines = ''.join(f.readlines()[:5])
                    if any(indicator in first_lines for indicator in [
                        '# This file is automatically generated',
                        '# Generated by',
                        '# Auto-generated',
                        '# Automatically created'
                    ]):
                        return True
            except (OSError, UnicodeDecodeError):
                pass
        
        return False
    
    def _is_critical_file(self, file_path: Path, rel_path: Path) -> bool:
        """Check if file is critical"""
        critical_files = {
            'setup.py', 'requirements.txt', 'pyproject.toml',
            'README.md', 'LICENSE', '.gitignore',
            'Makefile', 'Dockerfile', 'docker-compose.yml'
        }
        
        return file_path.name in critical_files
    
    def _is_working_file(self, file_path: Path, rel_path: Path) -> bool:
        """Check if file is actively worked on"""
        file_str = str(rel_path)
        
        # Recently modified in Git
        if file_str in self.git_info.get('recent_files', set()):
            return True
        
        # Recently modified on filesystem (within 7 days)
        try:
            mtime = file_path.stat().st_mtime
            seven_days_ago = time.time() - (7 * 24 * 3600)
            if mtime > seven_days_ago:
                return True
        except OSError:
            pass
        
        return False
    
    def _is_duplicate_file(self, file_path: Path, rel_path: Path) -> bool:
        """Check if file might be a duplicate"""
        # Look for similar named files
        stem = file_path.stem
        
        # Common duplicate patterns
        duplicate_indicators = [
            '_backup', '_old', '_copy', '_v2', '_v3',
            '_test', '_demo', '_temp', '_tmp'
        ]
        
        return any(indicator in stem for indicator in duplicate_indicators)
    
    def _path_to_module_name(self, rel_path: Path) -> str:
        """Convert file path to Python module name"""
        parts = list(rel_path.parts)
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        return '.'.join(parts)
    
    def _get_classification_reason(self, file_path: Path, rel_path: Path, classification: str) -> str:
        """Get reason for classification"""
        reasons = []
        
        if classification == 'GENERATED':
            if '__pycache__' in str(rel_path):
                reasons.append("Python cache file")
            elif file_path.suffix in {'.pyc', '.pyo'}:
                reasons.append("Compiled Python file")
            else:
                reasons.append("Generated file pattern")
        
        elif classification == 'CRITICAL':
            if str(rel_path) in self.imported_modules:
                reasons.append("Imported by other modules")
            elif file_path.name in {'setup.py', 'requirements.txt'}:
                reasons.append("Project configuration file")
        
        elif classification == 'WORKING':
            if str(rel_path) in self.git_info.get('recent_files', set()):
                reasons.append("Recently modified in Git")
            else:
                reasons.append("Active development file")
        
        elif classification == 'OBSOLETE':
            reasons.append("Not imported, not recently modified")
        
        elif classification == 'DUPLICATE':
            reasons.append("Potential duplicate based on naming")
        
        return '; '.join(reasons) if reasons else 'Automated classification'
    
    def _generate_analysis_report(self, analysis_time: float) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        total_files = sum(len(files) for files in self.analysis_results.values())
        total_size = sum(
            sum(f['size'] for f in files) 
            for files in self.analysis_results.values()
        )
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'project_root': str(self.project_root),
                'analysis_time_seconds': analysis_time,
                'total_files_analyzed': total_files,
                'total_size_mb': total_size / 1024 / 1024
            },
            'classification_summary': {
                category: {
                    'count': len(files),
                    'size_mb': sum(f['size'] for f in files) / 1024 / 1024,
                    'percentage': len(files) / total_files * 100 if total_files > 0 else 0
                }
                for category, files in self.analysis_results.items()
            },
            'detailed_results': self.analysis_results,
            'dependency_analysis': {
                'total_python_files': len(self.all_python_files),
                'total_imports': len(self.imported_modules),
                'import_graph_size': len(self.import_graph)
            },
            'git_analysis': {
                'recent_files_count': len(self.git_info.get('recent_files', set())),
                'tracked_files_count': len(self.git_info.get('tracked_files', set()))
            }
        }
        
        # Add recommendations
        report['cleanup_recommendations'] = self._generate_cleanup_recommendations()
        
        return report
    
    def _generate_cleanup_recommendations(self) -> Dict[str, Any]:
        """Generate cleanup recommendations"""
        recommendations = {
            'safe_to_delete': {
                'GENERATED': {
                    'count': len(self.analysis_results['GENERATED']),
                    'reason': 'Generated files can be safely deleted and regenerated',
                    'size_mb': sum(f['size'] for f in self.analysis_results['GENERATED']) / 1024 / 1024
                }
            },
            'review_for_deletion': {
                'OBSOLETE': {
                    'count': len(self.analysis_results['OBSOLETE']),
                    'reason': 'Files not imported and not recently modified - review before deletion',
                    'size_mb': sum(f['size'] for f in self.analysis_results['OBSOLETE']) / 1024 / 1024
                },
                'DUPLICATE': {
                    'count': len(self.analysis_results['DUPLICATE']),
                    'reason': 'Potential duplicates - manual review recommended',
                    'size_mb': sum(f['size'] for f in self.analysis_results['DUPLICATE']) / 1024 / 1024
                }
            },
            'keep_files': {
                'CRITICAL': {
                    'count': len(self.analysis_results['CRITICAL']),
                    'reason': 'Critical files - do not delete',
                    'size_mb': sum(f['size'] for f in self.analysis_results['CRITICAL']) / 1024 / 1024
                },
                'WORKING': {
                    'count': len(self.analysis_results['WORKING']),
                    'reason': 'Active working files - keep',
                    'size_mb': sum(f['size'] for f in self.analysis_results['WORKING']) / 1024 / 1024
                }
            },
            'needs_review': {
                'UNKNOWN': {
                    'count': len(self.analysis_results['UNKNOWN']),
                    'reason': 'Unknown classification - manual review required',
                    'size_mb': sum(f['size'] for f in self.analysis_results['UNKNOWN']) / 1024 / 1024
                }
            }
        }
        
        return recommendations
    
    def _save_analysis_results(self, report: Dict[str, Any]):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_file = self.project_root / f'file_analysis_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = self.project_root / f'file_analysis_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            self._write_summary_report(f, report)
        
        # Save cleanup script template
        script_file = self.project_root / f'cleanup_script_template_{timestamp}.sh'
        with open(script_file, 'w') as f:
            self._write_cleanup_script(f, report)
        
        print(f"   💾 Results saved:")
        print(f"      📊 Detailed report: {json_file}")
        print(f"      📄 Summary report: {summary_file}")
        print(f"      📝 Cleanup script: {script_file}")
    
    def _write_summary_report(self, f, report: Dict[str, Any]):
        """Write human-readable summary report"""
        f.write("="*80 + "\n")
        f.write("PROJECT FILE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        meta = report['analysis_metadata']
        f.write(f"Analysis Date: {meta['timestamp']}\n")
        f.write(f"Project Root: {meta['project_root']}\n")
        f.write(f"Analysis Time: {meta['analysis_time_seconds']:.2f} seconds\n")
        f.write(f"Total Files: {meta['total_files_analyzed']}\n")
        f.write(f"Total Size: {meta['total_size_mb']:.1f} MB\n\n")
        
        # Classification summary
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("-" * 40 + "\n")
        for category, info in report['classification_summary'].items():
            f.write(f"{category:12} {info['count']:5} files ({info['size_mb']:6.1f} MB, {info['percentage']:5.1f}%)\n")
        
        f.write("\n")
        
        # Cleanup recommendations
        f.write("CLEANUP RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        recs = report['cleanup_recommendations']
        
        f.write("\n🟢 SAFE TO DELETE:\n")
        for category, info in recs['safe_to_delete'].items():
            f.write(f"  {category}: {info['count']} files ({info['size_mb']:.1f} MB)\n")
            f.write(f"    → {info['reason']}\n")
        
        f.write("\n🟡 REVIEW FOR DELETION:\n")
        for category, info in recs['review_for_deletion'].items():
            f.write(f"  {category}: {info['count']} files ({info['size_mb']:.1f} MB)\n")
            f.write(f"    → {info['reason']}\n")
        
        f.write("\n🔴 KEEP FILES:\n")
        for category, info in recs['keep_files'].items():
            f.write(f"  {category}: {info['count']} files ({info['size_mb']:.1f} MB)\n")
            f.write(f"    → {info['reason']}\n")
        
        f.write("\n⚪ NEEDS REVIEW:\n")
        for category, info in recs['needs_review'].items():
            f.write(f"  {category}: {info['count']} files ({info['size_mb']:.1f} MB)\n")
            f.write(f"    → {info['reason']}\n")
        
        # Top files by category
        f.write("\n" + "="*80 + "\n")
        f.write("TOP FILES BY CATEGORY (First 10)\n")
        f.write("="*80 + "\n")
        
        for category, files in report['detailed_results'].items():
            if files:
                f.write(f"\n{category} FILES:\n")
                f.write("-" * len(category) + "-------\n")
                sorted_files = sorted(files, key=lambda x: x['size'], reverse=True)[:10]
                for file_info in sorted_files:
                    size_kb = file_info['size'] / 1024
                    f.write(f"  {file_info['path']:60} ({size_kb:6.1f} KB)\n")
                    if file_info.get('classification_reason'):
                        f.write(f"    Reason: {file_info['classification_reason']}\n")
    
    def _write_cleanup_script(self, f, report: Dict[str, Any]):
        """Write cleanup script template"""
        f.write("#!/bin/bash\n")
        f.write("# Cleanup Script Template - REVIEW BEFORE EXECUTING\n")
        f.write("# Generated from file analysis\n\n")
        
        f.write("set -e  # Exit on any error\n\n")
        
        f.write("# BACKUP FIRST (UNCOMMENT TO ENABLE)\n")
        f.write("# tar -czf project_backup_$(date +%Y%m%d_%H%M%S).tar.gz .\n\n")
        
        f.write("echo 'Starting cleanup...'\n\n")
        
        # Generated files (safe to delete)
        generated_files = report['detailed_results']['GENERATED']
        if generated_files:
            f.write("# GENERATED FILES (Safe to delete)\n")
            f.write("echo 'Cleaning generated files...'\n")
            for file_info in generated_files:
                f.write(f"# rm -rf '{file_info['path']}'  # {file_info['classification_reason']}\n")
            f.write("\n")
        
        # Obsolete files (review before deletion)
        obsolete_files = report['detailed_results']['OBSOLETE']
        if obsolete_files:
            f.write("# OBSOLETE FILES (Review before uncommenting)\n")
            f.write("echo 'Would clean obsolete files (currently commented out)...'\n")
            for file_info in obsolete_files[:20]:  # Limit to first 20
                f.write(f"# rm '{file_info['path']}'  # {file_info['classification_reason']}\n")
            f.write("\n")
        
        f.write("echo 'Cleanup complete!'\n")
        f.write("echo 'REMEMBER: Review this script before executing!'\n")


def main():
    """Main entry point"""
    project_root = "/Users/snehagupta/Model_Based_RL_for_Predictive_Human_Intent_Recognition/project2_human_intent_rl"
    
    analyzer = FileAnalysisSystem(project_root)
    report = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    
    for category, info in report['classification_summary'].items():
        print(f"{category:12} {info['count']:5} files ({info['size_mb']:6.1f} MB, {info['percentage']:5.1f}%)")
    
    print("\n🎯 CLEANUP POTENTIAL:")
    recs = report['cleanup_recommendations']
    
    safe_delete_size = sum(cat['size_mb'] for cat in recs['safe_to_delete'].values())
    review_delete_size = sum(cat['size_mb'] for cat in recs['review_for_deletion'].values())
    
    print(f"   🟢 Safe to delete: {safe_delete_size:.1f} MB")
    print(f"   🟡 Review for deletion: {review_delete_size:.1f} MB")
    print(f"   💾 Potential space savings: {safe_delete_size + review_delete_size:.1f} MB")
    
    print("\n⚠️  REMEMBER: This is analysis only. Review results before any cleanup!")
    print("="*80)


if __name__ == '__main__':
    main()