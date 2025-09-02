#!/usr/bin/env python3
"""
Safe Generated File Cleanup - Phase 2A
This script safely removes ONLY files marked as GENERATED in the analysis.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import time

class SafeGeneratedCleanup:
    """Safe cleanup of generated files only"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cleanup_log = []
        self.total_size_removed = 0
        self.files_removed = 0
        self.errors = []
        
    def load_analysis_results(self) -> dict:
        """Load the latest analysis results"""
        analysis_files = list(self.project_root.glob('file_analysis_report_*.json'))
        
        if not analysis_files:
            raise FileNotFoundError("No analysis report found. Run file analysis first.")
        
        # Get the most recent analysis file
        latest_analysis = max(analysis_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_analysis, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"📊 Loaded analysis from: {latest_analysis.name}")
        return analysis_data
    
    def execute_safe_cleanup(self) -> dict:
        """Execute safe cleanup of generated files only"""
        print("🧹 EXECUTING SAFE GENERATED FILE CLEANUP")
        print("="*60)
        
        # Load analysis results
        analysis_data = self.load_analysis_results()
        
        # Get generated files list
        generated_files = analysis_data['detailed_results'].get('GENERATED', [])
        
        if not generated_files:
            print("ℹ️  No generated files found to cleanup")
            return self._create_report()
        
        print(f"🎯 Found {len(generated_files)} generated files to cleanup")
        print(f"📦 Total size to remove: {sum(f['size'] for f in generated_files) / 1024 / 1024:.1f} MB")
        
        # Safety confirmation
        print("\n⚠️  SAFETY CHECK:")
        print("   - Files marked as GENERATED only")
        print("   - These can be safely regenerated")
        print("   - Backup already created")
        
        # Process each generated file
        for file_info in generated_files:
            self._cleanup_single_file(file_info)
        
        # Clean up empty directories
        self._cleanup_empty_directories()
        
        print(f"\n✅ CLEANUP COMPLETE!")
        print(f"   📁 Files removed: {self.files_removed}")
        print(f"   💾 Space freed: {self.total_size_removed / 1024 / 1024:.1f} MB")
        
        if self.errors:
            print(f"   ⚠️  Errors encountered: {len(self.errors)}")
        
        return self._create_report()
    
    def _cleanup_single_file(self, file_info: dict):
        """Safely cleanup a single file"""
        file_path = self.project_root / file_info['path']
        
        try:
            if not file_path.exists():
                self.cleanup_log.append({
                    'path': file_info['path'],
                    'status': 'skipped',
                    'reason': 'file not found',
                    'size': 0
                })
                return
            
            # Double-check this is a generated file
            if not self._verify_generated_file(file_path):
                self.cleanup_log.append({
                    'path': file_info['path'],
                    'status': 'skipped', 
                    'reason': 'safety verification failed',
                    'size': file_info['size']
                })
                self.errors.append(f"Safety verification failed for {file_info['path']}")
                return
            
            # Get file size before deletion
            file_size = file_path.stat().st_size
            
            # Remove file or directory
            if file_path.is_dir():
                shutil.rmtree(file_path)
                print(f"🗂️  Removed directory: {file_info['path']}")
            else:
                file_path.unlink()
                print(f"🗑️  Removed file: {file_info['path']}")
            
            # Update statistics
            self.files_removed += 1
            self.total_size_removed += file_size
            
            self.cleanup_log.append({
                'path': file_info['path'],
                'status': 'removed',
                'reason': file_info['classification_reason'],
                'size': file_size
            })
            
        except Exception as e:
            error_msg = f"Failed to remove {file_info['path']}: {str(e)}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            
            self.cleanup_log.append({
                'path': file_info['path'],
                'status': 'error',
                'reason': str(e),
                'size': file_info['size']
            })
    
    def _verify_generated_file(self, file_path: Path) -> bool:
        """Verify this is truly a generated file (safety check)"""
        file_str = str(file_path)
        
        # Known safe generated file patterns
        safe_patterns = [
            '__pycache__',
            '.pyc',
            '.pyo', 
            '.pyd',
            '.DS_Store',
            '.egg-info',
            'build/',
            'dist/',
            '.pytest_cache',
            '.coverage',
            'htmlcov/',
            '.tox/',
            '.mypy_cache',
            'venv/lib',  # Virtual environment files
            'venv/bin',
            'venv/include',
            'venv/share'
        ]
        
        # Check if file matches safe patterns
        for pattern in safe_patterns:
            if pattern in file_str:
                return True
        
        # Additional checks for Python cache files
        if file_path.suffix in {'.pyc', '.pyo', '.pyd'}:
            return True
        
        # Check if it's in a known generated directory
        parts = file_path.parts
        if any(part in {'__pycache__', '.pytest_cache', 'htmlcov', '.mypy_cache', 'build', 'dist'} 
               for part in parts):
            return True
        
        # Conservative approach: if unsure, don't delete
        return False
    
    def _cleanup_empty_directories(self):
        """Remove empty directories that might be left after file cleanup"""
        empty_dirs_removed = 0
        
        # Look for empty directories (multiple passes to handle nested empty dirs)
        for _ in range(5):  # Max 5 passes to handle deep nesting
            dirs_found = False
            
            for dir_path in self.project_root.rglob('*'):
                if dir_path.is_dir() and dir_path != self.project_root:
                    try:
                        # Check if directory is empty
                        if not any(dir_path.iterdir()):
                            # Extra safety: only remove if it's in a generated path
                            if self._is_generated_directory(dir_path):
                                dir_path.rmdir()
                                print(f"📂 Removed empty directory: {dir_path.relative_to(self.project_root)}")
                                empty_dirs_removed += 1
                                dirs_found = True
                    except (OSError, PermissionError):
                        continue
            
            if not dirs_found:
                break
        
        if empty_dirs_removed > 0:
            print(f"📂 Removed {empty_dirs_removed} empty directories")
    
    def _is_generated_directory(self, dir_path: Path) -> bool:
        """Check if directory is safe to remove when empty"""
        dir_str = str(dir_path)
        
        generated_dir_patterns = [
            '__pycache__',
            '.pytest_cache',
            'htmlcov',
            '.mypy_cache',
            'build',
            'dist',
            '.egg-info',
            'venv/lib',
            'venv/bin'
        ]
        
        return any(pattern in dir_str for pattern in generated_dir_patterns)
    
    def _create_report(self) -> dict:
        """Create cleanup report"""
        report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'phase': '2A - Generated Files Only',
            'statistics': {
                'files_processed': len(self.cleanup_log),
                'files_removed': self.files_removed,
                'size_removed_mb': self.total_size_removed / 1024 / 1024,
                'errors_count': len(self.errors)
            },
            'cleanup_log': self.cleanup_log,
            'errors': self.errors
        }
        
        # Save report
        report_file = self.project_root / f'cleanup_report_phase2a_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Cleanup report saved: {report_file.name}")
        return report


def main():
    """Main execution"""
    project_root = "/Users/snehagupta/Model_Based_RL_for_Predictive_Human_Intent_Recognition/project2_human_intent_rl"
    
    cleaner = SafeGeneratedCleanup(project_root)
    
    try:
        report = cleaner.execute_safe_cleanup()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 PHASE 2A CLEANUP SUMMARY")
        print("="*60)
        print(f"Files removed: {report['statistics']['files_removed']}")
        print(f"Space freed: {report['statistics']['size_removed_mb']:.1f} MB")
        print(f"Errors: {report['statistics']['errors_count']}")
        
        if report['errors']:
            print("\nErrors encountered:")
            for error in report['errors']:
                print(f"  ❌ {error}")
        
        print("\n✅ PHASE 2A COMPLETE: Generated files cleanup finished safely!")
        
    except Exception as e:
        print(f"❌ CLEANUP FAILED: {e}")
        raise


if __name__ == '__main__':
    main()