#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line script to generate comprehensive reports from CMB analysis results.

This script provides a simple interface to generate reports for different CMB analysis tests.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CMB-ReportGenerator")

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.reporting.report_generator import generate_report, ReportConfig


def main():
    """Main function to parse arguments and generate reports."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive reports from CMB analysis results"
    )
    
    parser.add_argument(
        "results_file",
        help="Path to the results file (pickle or JSON)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save the reports (defaults to same directory as results file)"
    )
    
    parser.add_argument(
        "--title", "-t",
        default="CMB Analysis Report",
        help="Report title"
    )
    
    parser.add_argument(
        "--author", "-a",
        default="Stephanie Alexander",
        help="Report author"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["all", "html", "markdown", "pdf"],
        default="all",
        help="Report format(s) to generate"
    )
    
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation"
    )
    
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip Markdown report generation"
    )
    
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation"
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.isfile(args.results_file):
        logger.error(f"Results file not found: {args.results_file}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.results_file))
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Configure report generation
    config = ReportConfig(
        title=args.title,
        author=args.author,
        include_html=not args.no_html and (args.format in ["all", "html"]),
        include_markdown=not args.no_markdown and (args.format in ["all", "markdown"]),
        include_pdf=not args.no_pdf and (args.format in ["all", "pdf"])
    )
    
    logger.info(f"Generating report for {args.results_file}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        output_files = generate_report(args.results_file, output_dir, config)
        
        logger.info("Report generation completed successfully")
        logger.info("Generated files:")
        for format_name, file_path in output_files.items():
            logger.info(f"  - {format_name}: {file_path}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
